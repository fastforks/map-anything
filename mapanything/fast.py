"""Fast inference wrapper for MapAnything: same `infer()` contract and outputs as `MapAnything.infer` with its defaults
(apply_mask, mask_edges, no confidence / multi-view masks), different precision / kernels / host work.

    from mapanything.fast import FastMapAnything
    fm = FastMapAnything(model, variant="friendly")   # converts encoder, info sharing and dense head in place
    preds = fm.infer(views)

Changes vs upstream `infer` (bf16 autocast, memory_efficient_inference=True):
- host work: no `torch.cuda.empty_cache()` / `mem_get_info()` per call (adaptive minibatch); the dense head runs in fixed
  view chunks (`dense_chunk`) instead; the edge mask (normals + depth edges, upstream CPU numpy, ~0.1 s per view) runs on
  the GPU with the same semantics (incl. numpy's NaN handling of arccos(dot > 1)); img_no_norm is computed on the GPU.
- encoder (DINOv2 ViT-g) and info sharing (16 alternating-attention blocks): pure fp16 weights + activations (upstream:
  bf16 autocast with an fp32 residual stream).
- dense DPT head: fp16 channels-last; the final 1x1 conv and the adaptors stay fp32. Pose / scale heads and the geometric
  input encoders stay fp32 (as upstream: autocast disabled there).
variant "friendly": plain PyTorch ops (SDPA attention, fp32-accumulate GEMMs / convs) -> ONNX / TensorRT exportable.
variant "max": + vendored fastbb flash attention (fp16 accumulate), fp16-accumulate cuBLAS GEMMs, fp16-accumulate cuDNN convs.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapanything.utils.geometry import quaternion_to_rotation_matrix, recover_pinhole_intrinsics_from_ray_directions
from mapanything.utils.inference import preprocess_input_views_for_inference, validate_input_views_for_inference
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
from uniception.models.prediction_heads.base import (AdaptorInput, PredictionHeadInput, PredictionHeadLayeredInput,
                                                     PredictionHeadTokenInput)

VARIANTS = {
    "friendly": dict(attn="sdpa", gacc=False, conv_hacc=False),
    "max": dict(attn="hh", gacc="now3", gacc_info=False, conv_hacc=False),  # fp16-acc info sharing moves the metric scale 5e-4
}
# gacc: False = fp32-accumulate GEMMs, True = fp16 accumulate (cuBLAS flag), "now3" = fp16 accumulate except the SwiGLU
# down projection w3 (K = 4096); gacc_enc / gacc_info override it for the encoder / the info-sharing blocks.


class _Conv1x1Fp32(nn.Module):
    """final DPT 1x1 conv in fp32 on a channels-last fp16 input: NHWC cast + fp32 GEMM (no NCHW transpose copy)"""

    def __init__(self, conv):
        super().__init__()
        self.weight = nn.Parameter(conv.weight.detach().float().flatten(1).clone(), requires_grad=False)
        self.bias = nn.Parameter(conv.bias.detach().float().clone(), requires_grad=False)

    def forward(self, x):
        y = F.linear(x.permute(0, 2, 3, 1).float(), self.weight, self.bias)  # [N, H, W, C_out]
        return y.permute(0, 3, 1, 2)


class FastMapAnything:
    def __init__(self, model, variant="friendly", dense_chunk=8, **over):
        from .fastbb_vendored.vit import FastViT

        dense_chunk = over.pop("dense_chunk", dense_chunk)
        cfg = dict(VARIANTS[variant], **over)
        self.m, self.variant, self.cfg, self.dense_chunk = model.eval(), variant, cfg, dense_chunk
        self.g_enc, self.g_info = cfg.get("gacc_enc", cfg["gacc"]), cfg.get("gacc_info", cfg["gacc"])
        self.stream32 = cfg.get("stream32", True)  # info-sharing residual stream fp32 (fp16 stream: metric scale off by ~5e-4)
        self.fv = FastViT(model.encoder.model, attn=cfg["attn"], gacc=self.g_enc is not False)
        self.fv.block = lambda b, x: self._block(b, x, self.g_enc)
        info = model.info_sharing
        info.half()
        if self.stream32:
            info.norm.float()  # final / intermediate norms read the fp32 stream
            for b in info.self_attention_blocks:  # LayerScale in fp32 as under upstream AMP (fp16 gamma: metric scale off ~5e-4)
                b.ls1.gamma.data, b.ls2.gamma.data = b.ls1.gamma.data.float(), b.ls2.gamma.data.float()
        self.blocks = list(info.self_attention_blocks)
        for b in self.blocks:
            assert b.custom_positional_encoding is None and isinstance(b.attn.q_norm, nn.Identity)
            b.forward = self._block_fwd(b)
        dh = model.dense_head
        last = dh[1].conv2[2]
        dh[1].conv2[2] = nn.Identity()
        dh.half().to(memory_format=torch.channels_last)
        assert last.kernel_size == (1, 1)
        dh[1].conv2[2] = _Conv1x1Fp32(last)  # final 1x1 conv stays fp32
        if cfg["conv_hacc"]:
            from .fastbb_vendored.cudnn_conv import patch_convs

            patch_convs(dh)
        model._encode_n_views = self._encode_n_views
        # geometric input encoders: upstream runs them on all-zero inputs and multiplies the result by a False mask when an
        # input is absent; skip them then (exact: adds 0)
        for name, keys, prob in (("_encode_and_fuse_ray_dirs", ("ray_directions_cam",), "ray_dirs_prob"),
                                 ("_encode_and_fuse_depths", ("depth_along_ray",), "depth_prob"),
                                 ("_encode_and_fuse_cam_quats_and_trans", ("camera_pose_quats", "camera_pose_trans"),
                                  "cam_prob")):
            setattr(model, name, self._skip_absent(getattr(model, name), keys, prob))
        model.downstream_head = self._downstream_head
        self.settings = (f"FastMapAnything variant={variant} {cfg} dense_chunk={dense_chunk} stream32={self.stream32}; fp16 encoder/info sharing/"
                         "DPT (last conv fp32), fp32 pose/scale heads + geometric encoders, GPU edge mask")

    # ------------------------------------------------------------------ model pieces
    def _block_fwd(self, b):
        fv = self.fv

        def fwd(x, xpos=None):
            from .fastbb_vendored.vit import fp16_accumulate

            with torch.autocast("cuda", enabled=False), fp16_accumulate(self.g_info is not False):
                return self._block(b, x.float() if self.stream32 else x.half(), self.g_info)

        return fwd

    def _block(self, b, x, gacc):
        """DINOv2 / uniception SelfAttentionBlock (LayerScale, SwiGLU or GELU MLP), fp16; caller sets the accumulate flag"""
        a = b.attn
        B, N, C = x.shape
        H = a.num_heads
        qkv = F.linear(b.norm1(x.half()), a.qkv.weight, a.qkv.bias).view(B, N, 3, H, C // H)
        o = F.linear(self.fv._attn(a, qkv, B, N, C, H), a.proj.weight, a.proj.bias)
        x = torch.addcmul(x, o, b.ls1.gamma)  # x fp32 (stream32: residual stream kept fp32 like upstream AMP) or fp16
        m = b.mlp
        if hasattr(m, "w12"):
            x1, x2 = F.linear(b.norm2(x.half()), m.w12.weight, m.w12.bias).chunk(2, dim=-1)
            h, w = F.silu(x1).mul_(x2), m.w3
        else:
            h, w = m.act(F.linear(b.norm2(x.half()), m.fc1.weight, m.fc1.bias)), m.fc2
        if gacc == "now3":
            torch.backends.cuda.matmul.allow_fp16_accumulation = False
            o = F.linear(h, w.weight, w.bias)
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
        else:
            o = F.linear(h, w.weight, w.bias)
        return x.addcmul_(o, b.ls2.gamma)

    def _skip_absent(self, fn, keys, prob):
        def f(views, num_views, batch_size_per_view, feats, *rest):
            c = self.m.geometric_input_config
            off = c["overall_prob"] <= 0 or c["dropout_prob"] >= 1 or c[prob] <= 0
            if off or not any(all(k in v for k in keys) for v in views):
                return feats
            return fn(views, num_views, batch_size_per_view, feats, *rest)

        return f

    def _encode_n_views(self, views):
        n = len(views)
        x = torch.cat([v["img"] for v in views], 0)
        r = self.fv.forward_features(x)
        B, _, H, W = x.shape
        ps = self.fv.ps
        feats = r["x_norm_patchtokens"].float().permute(0, 2, 1).reshape(B, -1, H // ps, W // ps).contiguous()
        regs = torch.cat([r["x_norm_clstoken"][:, None], r["x_norm_regtokens"]], 1).float().permute(0, 2, 1).contiguous()
        return feats.chunk(n, 0), (regs.chunk(n, 0) if self.m.use_register_tokens_from_encoder else None)

    def _downstream_head(self, dense_head_inputs, scale_head_inputs, img_shape, memory_efficient_inference=False,
                         minibatch_size=None):
        m = self.m
        assert m.pred_head_type == "dpt+pose"
        feats = [f.half().contiguous(memory_format=torch.channels_last) for f in dense_head_inputs]
        N = feats[0].shape[0]
        c = self.dense_chunk or N
        outs = []
        for s in range(0, N, c):
            with torch.autocast("cuda", enabled=False):
                ho = m.dense_head(PredictionHeadLayeredInput(list_features=[f[s:s + c] for f in feats],
                                                             target_output_shape=img_shape))
            outs.append(ho.decoded_channels.float())
        dense = m.dense_adaptor(AdaptorInput(adaptor_feature=torch.cat(outs, 0) if len(outs) > 1 else outs[0],
                                             output_shape_hw=img_shape))
        ph = m.pose_head(PredictionHeadInput(last_feature=dense_head_inputs[-1].float()))
        pose = m.pose_adaptor(AdaptorInput(adaptor_feature=ph.decoded_channels, output_shape_hw=img_shape))
        sh = m.scale_head(PredictionHeadTokenInput(last_feature=scale_head_inputs.float()))
        scale = m.scale_adaptor(AdaptorInput(adaptor_feature=sh.decoded_channels, output_shape_hw=img_shape))
        return dense, pose, scale.value.squeeze(-1)

    # ------------------------------------------------------------------ infer
    @torch.inference_mode()
    def infer(self, views, ignore_calibration_inputs=False, ignore_depth_inputs=False, ignore_pose_inputs=False,
              ignore_depth_scale_inputs=False, ignore_pose_scale_inputs=False, edge_normal_threshold=5.0,
              edge_depth_threshold=0.03):
        m = self.m
        views = validate_input_views_for_inference(views)
        skip = {"instance", "idx", "true_shape", "data_norm_type"}
        for v in views:
            for k in v:
                if k not in skip and hasattr(v[k], "to"):
                    v[k] = v[k].to(m.device, non_blocking=True)
        views = preprocess_input_views_for_inference(views)
        m._configure_geometric_input_config(use_calibration=not ignore_calibration_inputs, use_depth=not ignore_depth_inputs,
                                            use_pose=not ignore_pose_inputs, use_depth_scale=not ignore_depth_scale_inputs,
                                            use_pose_scale=not ignore_pose_scale_inputs)
        with torch.autocast("cuda", enabled=False):
            preds = m.forward(views, memory_efficient_inference=False)
        preds = postprocess_gpu(preds, views, edge_normal_threshold, edge_depth_threshold)
        m._restore_original_geometric_input_config()
        return preds


# ====================================================================== GPU post-processing
def _shift(x, dy, dx):
    """x [N, H+2, W+2, ...] padded -> view of the [N, H, W, ...] window shifted by (dy, dx) in {-1, 0, 1}"""
    H, W = x.shape[1] - 2, x.shape[2] - 2
    return x[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]


def _unit(v):
    return v / (torch.linalg.vector_norm(v, dim=-1, keepdim=True) + 1e-12)


def _maxpool_nan(x):
    """3x3 / stride 1 max pool ignoring NaN (numpy nanmax with NaN padding); all-NaN windows -> -inf"""
    x = torch.where(torch.isnan(x), float("-inf"), x)
    return F.max_pool2d(x[:, None], 3, 1, 1)[:, 0]


def edge_mask_gpu(pts, depth, mask, normal_tol=5.0, depth_rtol=0.03):
    """= ~(depth_edge(depth, rtol, mask) & normals_edge(points_to_normals(pts, mask), tol)) of mapanything.utils.geometry,
    batched: pts [N, H, W, 3] fp32, depth [N, H, W], mask [N, H, W] bool -> [N, H, W] bool."""
    pts = pts.float()
    p = F.pad(pts.permute(0, 3, 1, 2), (1, 1, 1, 1)).permute(0, 2, 3, 1)  # zero padding
    mp = F.pad(mask[:, None].float(), (1, 1, 1, 1))[:, 0] > 0.5           # False padding
    c = _shift(p, 0, 0)
    up, left, down, right = (_shift(p, -1, 0) - c, _shift(p, 0, -1) - c, _shift(p, 1, 0) - c, _shift(p, 0, 1) - c)
    mc = _shift(mp, 0, 0)
    mu, ml, md, mr = _shift(mp, -1, 0), _shift(mp, 0, -1), _shift(mp, 1, 0), _shift(mp, 0, 1)
    normal = torch.zeros_like(pts)
    nmask = torch.zeros_like(mask)
    for a, b, va in ((up, left, mu & ml), (left, down, ml & md), (down, right, md & mr), (right, up, mr & mu)):
        va = va & mc
        normal += _unit(torch.linalg.cross(a, b, dim=-1)) * va[..., None]
        nmask |= va
    normal = _unit(normal)
    normal = torch.where(nmask[..., None], normal, 0.0)
    # normals_edge: replicate padding; the upstream mask window is transposed relative to the normal window
    # (sliding_window_2d over axes (-3, -2) of a 2-D mask -> axes (1, 0)), reproduced here
    n = _unit(normal)
    npad = F.pad(n.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate").permute(0, 2, 3, 1)
    mpad = F.pad(nmask[:, None].float(), (1, 1, 1, 1), mode="replicate")[:, 0] > 0.5
    ang = None
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            dot = (n * _shift(npad, a, b)).sum(-1)
            v = torch.where(_shift(mpad, b, a), torch.arccos(dot), 0.0)
            ang = v if ang is None else torch.maximum(ang, v)  # torch.maximum propagates NaN like np.max
    normal_edges = _maxpool_nan(ang) > math.radians(normal_tol)
    d = depth.float()
    inf = torch.tensor(float("-inf"), device=d.device)
    diff = (F.max_pool2d(torch.where(mask, d, inf)[:, None], 3, 1, 1) +
            F.max_pool2d(torch.where(mask, -d, inf)[:, None], 3, 1, 1))[:, 0]
    depth_edges = diff / d > depth_rtol
    return ~(depth_edges & normal_edges)


def postprocess_gpu(raw_outputs, views, edge_normal_threshold=5.0, edge_depth_threshold=0.03):
    """= postprocess_model_outputs_for_inference with its infer() defaults (apply_mask, mask_edges, no confidence /
    multi-view masks), all on the GPU, edge masks batched over the views."""
    out = []
    for raw, v in zip(raw_outputs, views):
        p = dict(raw)
        img = v["img"]
        nt = v["data_norm_type"][0]
        if nt in IMAGE_NORMALIZATION_DICT:
            nrm = IMAGE_NORMALIZATION_DICT[nt]
            mean, std = nrm.mean.to(img.device), nrm.std.to(img.device)
        else:
            mean, std = torch.zeros(3, device=img.device), torch.ones(3, device=img.device)
        p["img_no_norm"] = (img.permute(0, 2, 3, 1).float() * std + mean).clamp(0, 1)
        if "pts3d_cam" in p:
            p["depth_z"] = p["pts3d_cam"][..., 2:3]
        if "ray_directions" in p:
            p["intrinsics"] = recover_pinhole_intrinsics_from_ray_directions(p["ray_directions"])
        if "cam_trans" in p and "cam_quats" in p:
            B = p["cam_trans"].shape[0]
            T = torch.eye(4, device=img.device).repeat(B, 1, 1)
            T[:, :3, :3] = quaternion_to_rotation_matrix(p["cam_quats"])
            T[:, :3, 3] = p["cam_trans"]
            p["camera_poses"] = T
        out.append(p)
    if "non_ambiguous_mask" not in out[0]:
        return out
    sizes = [p["pts3d"].shape[0] for p in out]
    mask = torch.cat([p["non_ambiguous_mask"] for p in out], 0).bool()
    if "pts3d" in out[0]:
        em = edge_mask_gpu(torch.cat([p["pts3d"] for p in out], 0), torch.cat([p["depth_z"][..., 0] for p in out], 0),
                           mask, edge_normal_threshold, edge_depth_threshold)
        mask = mask & em
    for p, fm in zip(out, mask.split(sizes, 0)):
        fm = fm[..., None]
        for k in ("pts3d", "pts3d_cam", "depth_along_ray", "depth_z"):
            if k in p:
                p[k] = p[k] * fm
        p["mask"] = fm
    return out
