# vendored from fastforks/fastbb: fastbb/vit.py @ a61ba9c -- do not edit here, see fastbb/VENDORED.md
"""Fast inference for DINOv2-style ViTs (official dinov2 package, the Depth Anything V2 copy vendored by FoundationStereo,
DEFOM, MonSter(++), PromptStereo, Stereo Anywhere, WAFT, MoGe, DAv2, ...): a drop-in for `get_intermediate_layers`.

General (any input size that is a multiple of the patch size; nothing compiled or tuned per size):
- pure fp16 weights + activations, autocast disabled inside (callers running under bf16/fp16 autocast get fp16 tokens, or
  `out_dtype`), fp16-accumulate cuBLAS GEMMs (2x tensor-core rate on GeForce; ViT-L 6k tokens: 1.55x per GEMM)
- attention: prebuilt flash kernel for head_dim 64 reading q/k/v straight from the packed qkv GEMM output and writing
  [B, N, H*64] for the projection (no transposes / copies); `attn="fh"` = fp32 Q K^T + fp16-accumulate P V per 64-key block
  (error ~ SDPA's), `"hh"` = fp16 accumulate for both (fastest; validate per model), `"sdpa"` = torch SDPA (fallback)
- residual + LayerScale as one in-place addcmul per branch
- positional-embedding interpolation (bicubic, fp32, upstream formula incl. interpolate_offset / antialias) cached per grid
The block math is the upstream DINOv2 Block (eval, no drop path); only the precision / kernel choice changes.

    from fastbb.vit import FastViT
    fv = FastViT(model.pretrained, attn="fh")       # converts the ViT to fp16 in place
    fv.install()                                    # model.pretrained.get_intermediate_layers -> fast version
"""
import contextlib
import math

import torch
import torch.nn.functional as F

from .attention import attention


@contextlib.contextmanager
def fp16_accumulate(enabled=True):
    old = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = bool(enabled)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old


SMALL_WORK = 2.2e5  # B*N*C below this: torch SDPA (launch ~20 us CPU vs ~35 us for the ctypes kernel launch; such calls are
# CPU-bound). Measured b1 224: ViT-S/14, ViT-B/14, ViT-B/16 faster with SDPA; ViT-L/14 (257*1024) faster with flash64.


class FastViT:
    def __init__(self, vit, attn="fh", gacc=True, dtype=torch.float16, out_dtype=None, graphs=False, graph_noclone=False):
        self.pos_fp32 = vit.pos_embed.detach().float().clone()  # before the dtype cast (official interpolates the fp32 table)
        self.vit = vit.to(dtype).eval()
        self.dtype, self.attn, self.gacc, self.out_dtype = dtype, attn, gacc, out_dtype
        self.ps = vit.patch_size if isinstance(vit.patch_size, int) else vit.patch_size[0]
        self.nreg = getattr(vit, "num_register_tokens", 0) or 0
        self._pos = {}
        self._pos_version = self._version((vit.pos_embed,))
        self._graph_version = None
        blocks = []
        for b in vit.blocks:
            blocks.extend(list(b) if isinstance(b, torch.nn.ModuleList) or b.__class__.__name__ == "BlockChunk" else [b])
        self.blocks = [b for b in blocks if not isinstance(b, torch.nn.Identity)]
        self._orig = {}
        self.graphs, self.graph_noclone, self._graphs, self._pool = graphs, graph_noclone, {}, None

    @staticmethod
    def _version(parameters):
        try:
            return tuple((id(p), p._version, p.device, p.dtype) for p in parameters)
        except RuntimeError:  # inference tensors do not track mutations
            return None

    def _refresh_pos(self):
        version = self._version((self.vit.pos_embed,))
        if version is None or version != self._pos_version:
            if version is not None and self._pos_version is not None and version[0][:2] == self._pos_version[0][:2]:
                self.pos_fp32 = self.pos_fp32.to(self.vit.pos_embed.device)
            else:
                self.pos_fp32 = self.vit.pos_embed.detach().float().clone()
            self._pos.clear()
            self._graphs.clear()
            self._pool = None
            self._pos_version = version

    # ---------------------------------------------------------------- tokens
    def pos_embed(self, h0, w0):
        self._refresh_pos()
        key = (h0, w0)
        if key not in self._pos:
            vit, pe = self.vit, self.pos_fp32
            N = pe.shape[1] - 1
            M = int(math.sqrt(N))
            if N == h0 * w0 and h0 == w0:
                out = pe
            else:
                off = getattr(vit, "interpolate_offset", 0.1)
                kw = dict(scale_factor=((h0 + off) / M, (w0 + off) / M)) if off else dict(size=(h0, w0))
                p = F.interpolate(pe[:, 1:].reshape(1, M, M, -1).permute(0, 3, 1, 2), mode="bicubic",
                                  antialias=getattr(vit, "interpolate_antialias", False), **kw)
                assert p.shape[-2:] == (h0, w0)
                out = torch.cat([pe[:, :1], p.permute(0, 2, 3, 1).flatten(1, 2)], 1)
            self._pos[key] = out.to(self.dtype)
        return self._pos[key]

    def tokens(self, x):
        vit = self.vit
        B, _, h, w = x.shape
        t = vit.patch_embed(x.to(self.dtype))
        t = torch.cat([vit.cls_token.to(self.dtype).expand(B, -1, -1), t], 1)
        t = t + self.pos_embed(h // self.ps, w // self.ps)
        if self.nreg:
            t = torch.cat([t[:, :1], vit.register_tokens.to(self.dtype).expand(B, -1, -1), t[:, 1:]], 1)
        return t

    # ---------------------------------------------------------------- block
    def block(self, b, x):
        if self.gacc == "chunk":
            return self.block_chunked(b, x)
        a = b.attn
        B, N, C = x.shape
        H = a.num_heads
        qkv = F.linear(b.norm1(x), a.qkv.weight, a.qkv.bias).view(B, N, 3, H, C // H)
        o = self._attn(a, qkv, B, N, C, H)
        o = F.linear(o, a.proj.weight, a.proj.bias)
        g1 = getattr(getattr(b, "ls1", None), "gamma", None)
        x = torch.addcmul(x, o, g1) if g1 is not None else x + o
        m = b.mlp
        if hasattr(m, "fc1"):
            act = getattr(m, "act", None)
            erf = act is None or (isinstance(act, torch.nn.GELU) and act.approximate == "none")
            if erf and self.gacc is False and self.attn != "sdpa" and m.fc1.bias is not None:
                # fp32-acc GEMM: cuBLASLt GELU epilogue saves the GELU pass (ViT-L fc1 6k tokens 1.32 -> 1.14 ms, same error;
                # with fp16 accumulation no gain -> plain path). Not in the friendly (sdpa) tier: private aten op.
                y = b.norm2(x)
                h = torch._addmm_activation(m.fc1.bias, y.reshape(-1, y.shape[-1]), m.fc1.weight.t(), use_gelu=True)
                h = h.view(*y.shape[:-1], -1)
            else:
                h = F.linear(b.norm2(x), m.fc1.weight, m.fc1.bias)
                h = F.gelu(h) if erf else act(h)
            if self.gacc == "nofc2":  # fc2 (K = 4 C) in fp32 accumulation, the other GEMMs (K = C) fp16
                torch.backends.cuda.matmul.allow_fp16_accumulation = False
                o = F.linear(h, m.fc2.weight, m.fc2.bias)
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            else:
                o = F.linear(h, m.fc2.weight, m.fc2.bias)
        else:  # SwiGLU FFN (ViT-g): w12 -> silu(x1) * x2 -> w3
            x12 = F.linear(b.norm2(x), m.w12.weight, m.w12.bias)
            x1, x2 = x12.chunk(2, dim=-1)
            o = F.linear(F.silu(x1).mul_(x2), m.w3.weight, m.w3.bias)
        g2 = getattr(getattr(b, "ls2", None), "gamma", None)
        return x.addcmul_(o, g2) if g2 is not None else x.add_(o)

    def _attn(self, a, qkv, B, N, C, H):
        if self.attn != "sdpa" and C // H == 64 and B * N * C >= SMALL_WORK:
            q, k, v = qkv.unbind(2)                                  # [B, N, H, 64] strided views
            return attention(q, k, v, scale=getattr(a, "scale", None), variant=self.attn).view(B, N, C)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        return F.scaled_dot_product_attention(q, k, v, scale=getattr(a, "scale", None)).transpose(1, 2).reshape(B, N, C)

    def block_chunked(self, b, x):
        """g3: all GEMMs through fastbb.gemm (chunked fp16 accumulation, error ~ fp32 accumulate) with fused epilogues:
        proj + LayerScale + residual, fc1 + GELU, fc2 + LayerScale + residual."""
        from .gemm import linear
        a = b.attn
        B, N, C = x.shape
        H = a.num_heads
        qkv = linear(b.norm1(x), a.qkv.weight, a.qkv.bias).view(B, N, 3, H, C // H)
        o = self._attn(a, qkv, B, N, C, H)
        x = linear(o, a.proj.weight, a.proj.bias, res=x, gamma=getattr(getattr(b, "ls1", None), "gamma", None))
        m = b.mlp
        if hasattr(m, "fc1"):
            act = getattr(m, "act", None)
            if act is None or (isinstance(act, torch.nn.GELU) and act.approximate == "none"):
                h = linear(b.norm2(x), m.fc1.weight, m.fc1.bias, act="gelu")
            else:
                h = act(linear(b.norm2(x), m.fc1.weight, m.fc1.bias))
            return linear(h, m.fc2.weight, m.fc2.bias, res=x, gamma=getattr(getattr(b, "ls2", None), "gamma", None))
        x12 = linear(b.norm2(x), m.w12.weight, m.w12.bias)
        x1, x2 = x12.chunk(2, dim=-1)
        return linear(F.silu(x1).mul_(x2), m.w3.weight, m.w3.bias, res=x, gamma=getattr(getattr(b, "ls2", None), "gamma", None))

    # ---------------------------------------------------------------- API
    def run(self, x, layers, norm=True):
        """-> list of tokens [B, 1 + nreg + HW, C] after the blocks in `layers` (normed if norm)."""
        out = []
        last = max(layers)
        with torch.autocast("cuda", enabled=False), fp16_accumulate(self.gacc is True or self.gacc == "nofc2"):
            t = self.tokens(x)
            for i, b in enumerate(self.blocks):
                t = self.block(b, t)
                if i in layers:
                    out.append(self.vit.norm(t) if norm else t.clone())
                if i >= last:
                    break
        return out

    def run_graphed(self, x, layers, norm=True):
        """Opt-in (graphs=True): one CUDA graph per (input shape, layers), captured on first use (~ms, no compile / tuning);
        removes the per-kernel CPU launch cost that dominates small inputs (ViT-S/B at <= 518 px). Outputs are the graph's
        static buffers: valid until the next call with the same shape (cloned here for safety unless graph_noclone)."""
        self._refresh_pos()
        version = self._version(self.vit.parameters())
        if version is None:
            return self.run(x, layers, norm)
        if version != self._graph_version:
            self._graphs.clear()
            self._pool = None
            self._graph_version = version
        key = (tuple(x.shape), x.dtype, x.device, tuple(sorted(layers)), norm)
        g = self._graphs.get(key)
        if g is None:
            xs = x.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.run(xs, layers, norm)  # warm-up: pos-embed cache, cuBLAS handles, kernel module
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                outs = self.run(xs, layers, norm)
            self._pool = graph.pool()
            g = self._graphs[key] = (graph, xs, outs)
        graph, xs, outs = g
        xs.copy_(x)
        graph.replay()
        return outs if self.graph_noclone else [o.clone() for o in outs]

    def get_intermediate_layers(self, x, n=1, reshape=False, return_class_token=False, norm=True):
        """Same contract as DinoVisionTransformer.get_intermediate_layers."""
        nb = len(self.blocks)
        layers = list(range(nb - n, nb)) if isinstance(n, int) else list(n)
        outs = (self.run_graphed if self.graphs else self.run)(x, set(layers), norm)
        od = self.out_dtype
        if od is not None and od != self.dtype:
            outs = [o.to(od) for o in outs]
        cls = [o[:, 0] for o in outs]
        outs = [o[:, 1 + self.nreg:] for o in outs]
        if reshape:
            B, _, h, w = x.shape
            outs = [o.reshape(B, h // self.ps, w // self.ps, -1).permute(0, 3, 1, 2).contiguous() for o in outs]
        if return_class_token:
            return tuple(zip(outs, cls))
        return tuple(outs)

    def forward_features(self, x):
        """Same keys as the official DinoVisionTransformer.forward_features (eval, no masks)."""
        t = (self.run_graphed if self.graphs else self.run)(x, {len(self.blocks) - 1}, norm=False)[0]
        with torch.autocast("cuda", enabled=False):
            n = self.vit.norm(t)
        r = 1 + self.nreg
        return {"x_norm_clstoken": n[:, 0], "x_norm_regtokens": n[:, 1:r], "x_norm_patchtokens": n[:, r:],
                "x_prenorm": t, "masks": None}

    def __call__(self, x):
        """= official forward (eval, is_training=False): head(x_norm_clstoken), head = Identity for the backbones."""
        return self.forward_features(x)["x_norm_clstoken"]

    def install(self):
        """Route the wrapped module's get_intermediate_layers through this object (instance attribute)."""
        self._orig["get_intermediate_layers"] = self.vit.__dict__.get("get_intermediate_layers")
        self.vit.get_intermediate_layers = self.get_intermediate_layers
        self.vit._fastbb = self
        return self


# ==================================================================== block-level drop-in (any ViT-style model)
class _BlockCfg:
    """Holds attn / gacc for the shared block math (borrows FastViT's methods)."""
    block = FastViT.block
    _attn = FastViT._attn
    block_chunked = FastViT.block_chunked

    def __init__(self, attn, gacc):
        self.attn, self.gacc = attn, gacc


def _ident(m):
    return m is None or isinstance(m, (torch.nn.Identity, torch.nn.Dropout))


def _block_supported(b):
    """Pre-norm ViT block: norm1 -> attn(qkv packed, proj) -> ls1 -> +x ; norm2 -> mlp(fc1, act, fc2 | w12, w3) -> ls2 -> +x
    (official DINOv2 / DAv2 copies / timm VisionTransformer Block). Anything else (q/k norm, gates, rope, masks) -> False."""
    a, m = getattr(b, "attn", None), getattr(b, "mlp", None)
    if a is None or m is None or not all(hasattr(b, n) for n in ("norm1", "norm2")):
        return False
    if not (isinstance(getattr(a, "qkv", None), torch.nn.Linear) and isinstance(getattr(a, "proj", None), torch.nn.Linear)):
        return False
    if not all(_ident(getattr(a, n, None)) for n in ("q_norm", "k_norm", "norm")) or getattr(a, "gate", None) is not None:
        return False
    if getattr(a, "qkv_bias_separate", False) or getattr(a, "rope", None) is not None:
        return False
    for n in ("drop_path", "drop_path1", "drop_path2"):
        if hasattr(b, n) and not _ident(getattr(b, n)) and type(getattr(b, n)).__name__ != "DropPath":
            return False
    for n in ("ls1", "ls2"):
        ls = getattr(b, n, None)
        if not (_ident(ls) or hasattr(ls, "gamma")):
            return False
    if hasattr(m, "fc1"):
        return _ident(getattr(m, "norm", None)) and isinstance(m.fc1, torch.nn.Linear) and isinstance(m.fc2, torch.nn.Linear)
    return hasattr(m, "w12") and hasattr(m, "w3")  # DINOv2 SwiGLU (ViT-g)


def _fast_block_forward(self, x, *args, **kw):
    if (args and args[0] is not None) or kw.get("attn_mask") is not None or kw.get("attn_bias") is not None \
            or kw.get("is_causal") or self.training:
        return type(self).forward(self, x, *args, **kw)
    cfg = self._fbb_cfg
    dt = x.dtype
    with torch.autocast("cuda", enabled=False), fp16_accumulate(cfg.gacc is True or cfg.gacc == "nofc2"):
        wd = self.attn.qkv.weight.dtype
        y = cfg.block(self, x.to(wd).contiguous())
    return y if dt == wd else y.to(dt)




def install_vit_blocks(model, spec="fh-g0", convert=True):
    """Patch every supported transformer block in `model` (forward -> fastbb block: packed qkv, flash64 attention for head_dim
    64 else SDPA, fp16(-acc) GEMMs, in-place LayerScale residual). Tokenizer, pos-embed, heads stay the model's own code.
    convert=True: model.half() (weights fp16; the patched blocks take / return the caller's dtype). Returns #patched blocks.
    Works for timm VisionTransformer (ViT / DeiT / DINOv2 / SigLIP-style blocks), official DINOv2, DAv2 copies."""
    from . import parse_vit_spec
    attn, gacc = parse_vit_spec(spec)
    if convert:
        model.half()
    n = 0
    for b in model.modules():
        if b is model and not hasattr(b, "norm1"):
            continue
        if _block_supported(b) and "_fbb_cfg" not in b.__dict__:
            b._fbb_cfg = _BlockCfg(attn, gacc)
            b.forward = _fast_block_forward.__get__(b)
            n += 1
    return n
