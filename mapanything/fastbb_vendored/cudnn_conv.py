# vendored from fastforks/fastbb: fastbb/cudnn_conv.py @ a61ba9c -- do not edit here, see fastbb/VENDORED.md
"""(copied from exp 004 mono_fast/cudnn_conv.py, originally exp 003 stereoanywhere) General fp16-accumulate convolutions through cuDNN graphs (port of RAFT-Stereo exp 002 `general.py` convs).

PyTorch always asks cuDNN for fp32 accumulation; on GeForce cards that runs the tensor cores at half rate
(RTX 5060 Ti: ~47 vs ~80 TFLOPS). Here every fp16 Conv2d/Conv3d (groups 1, dilation 1) runs as a plain cuDNN
conv_fprop graph with compute_data_type=HALF, NHWC / NDHWC in and out.

General = any input size, no per-size compilation or tuning: plans that need runtime compilation (NVRTC) are
deselected, the first precompiled plan in cuDNN's heuristic order is used (no timing), ~ms per new geometry.
No fused epilogue (those only exist as runtime-compiled kernels on sm_120): bias is added by PyTorch.
Engines 0 (reference, very slow) and 8 (crashes on sm_120, exp 002 DESIGN_FLAWS D9) are skipped.
FASTBB_CONV_VERIFY=1 checks every new plan against an fp32 reference (debug).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cudnn
except ImportError:  # optional: pip install nvidia-cudnn-frontend
    cudnn = None

# eng0: reference (very slow); eng8: crashes on sm_120 (exp 002 D9); eng28: implicit_convolve_hhgemm, no tensor cores,
# ranked first by the heuristic for (5,1) convs on sm_120 (9x slower than eng22, measured for CREStereo GRU shapes)
DENY = ("eng0", "eng8", "eng28")
VERIFY = os.environ.get("FASTBB_CONV_VERIFY") == "1"
_handle = None
_plans = {}
_ws = [None]
stats = {"cudnn": 0, "torch": 0, "built": 0}


def _get_handle():
    global _handle
    if _handle is None:
        _handle = cudnn.create_handle()
    return _handle


def _out_size(x, w, padding, stride):
    return [(x.shape[2 + i] + 2 * padding[i] - w.shape[2 + i]) // stride[i] + 1 for i in range(w.dim() - 2)]


def _cl_stride(shape):
    """channels-last strides for [N, C, *spatial]."""
    n, c, *sp = shape
    st = [0] * len(shape)
    acc = c
    st[1] = 1
    for i in range(len(sp) - 1, -1, -1):
        st[2 + i] = acc
        acc *= sp[i]
    st[0] = acc
    return st


def _build(x, w, padding, stride, ystride=None):
    H16 = cudnn.data_type.HALF
    nd = w.dim() - 2
    out_sp = _out_size(x, w, padding, stride)
    yshape = [x.shape[0], w.shape[0], *out_sp]
    h = _get_handle()
    try:
        g = cudnn.pygraph(io_data_type=H16, intermediate_data_type=H16, compute_data_type=H16, handle=h)
        X = g.tensor(name="X", dim=list(x.shape), stride=list(x.stride()), data_type=H16)
        W = g.tensor(name="W", dim=list(w.shape), stride=list(w.stride()), data_type=H16)
        Y = g.conv_fprop(image=X, weight=W, padding=list(padding), stride=list(stride), dilation=[1] * nd,
                         compute_data_type=H16)
        Y.set_output(True).set_data_type(H16).set_dim(yshape).set_stride(list(ystride) if ystride else _cl_stride(yshape))
        g.validate()
        g.build_operation_graph()
        g.deselect_behavior_notes([cudnn.behavior_note.RUNTIME_COMPILATION])
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
    except Exception:  # noqa: BLE001 - no half-compute graph for this geometry
        return None
    for i in range(g.get_execution_plan_count()):
        if g.get_plan_name_at_index(i).split("_")[0] in DENY:
            continue
        try:
            g.build_plan_at_index(i)
        except Exception:  # noqa: BLE001
            continue
        if VERIFY and not _verify(g, X, W, Y, i, x, w, yshape, padding, stride):
            continue
        stats["built"] += 1
        return g, X, W, Y, i, g.get_workspace_size_plan_at_index(i), yshape
    return None


def _verify(g, X, W, Y, i, x, w, yshape, padding, stride):
    xr = (torch.rand(x.shape, device=x.device) * 2 - 1).half().contiguous(memory_format=_mf(x.dim()))
    y = torch.full(yshape, float("nan"), device=x.device, dtype=torch.half).contiguous(memory_format=_mf(x.dim()))
    ws = torch.empty(max(g.get_workspace_size_plan_at_index(i), 1), device=x.device, dtype=torch.uint8)
    g.execute_plan_at_index({X: xr, W: w, Y: y}, ws, i, handle=_get_handle())
    conv = F.conv2d if x.dim() == 4 else F.conv3d
    ref = conv(xr.float(), w.float(), None, stride, padding)
    dev = float((y.float() - ref).abs().max() / (ref.abs().max() + 1e-3))
    ok = dev == dev and dev < 0.05
    if not ok:
        print(f"[cudnn_conv] rejected {g.get_plan_name_at_index(i)} x{tuple(x.shape)} w{tuple(w.shape)}: {dev:.3g}")
    return ok


def _mf(dim):
    return torch.channels_last if dim == 4 else torch.channels_last_3d


def conv_half_acc(x, w, b, padding, stride, out=None):
    """fp16 conv with fp16 accumulation; x any layout (made channels-last), w channels-last fp16. Returns channels-last,
    or writes into `out` (any strides, e.g. a channel slice of a channels-last buffer) and returns it."""
    mf = _mf(x.dim())
    x = x.contiguous(memory_format=mf)
    key = (tuple(x.shape), tuple(w.shape), tuple(padding), tuple(stride), x.device.index,
           tuple(out.stride()) if out is not None else None)
    e = _plans.get(key, False)
    if e is False:
        e = _plans[key] = _build(x, w, padding, stride, out.stride() if out is not None else None)
    if e is None:
        stats["torch"] += 1
        conv = F.conv2d if x.dim() == 4 else F.conv3d
        y = conv(x, w, b, stride, padding)
        if out is not None:
            return out.copy_(y)
        return y.contiguous(memory_format=mf)
    stats["cudnn"] += 1
    g, X, W, Y, i, ws_size, yshape = e
    if out is not None:
        assert list(out.shape) == list(yshape)
        y = out
    else:
        y = torch.empty(yshape, device=x.device, dtype=x.dtype, memory_format=mf)
    ws = _ws[0]
    if ws is None or ws.numel() < ws_size:
        ws = _ws[0] = torch.empty(max(ws_size, 1), device=x.device, dtype=torch.uint8)
    h = _get_handle()
    cudnn.set_stream(handle=h, stream=torch.cuda.current_stream().cuda_stream)
    g.execute_plan_at_index({X: x, W: w, Y: y}, ws, i, handle=h)
    if b is not None:
        y.add_(b.view(1, -1, *([1] * (x.dim() - 2))))
    return y


def _build_dgrad(x, w, padding, stride, yshape):
    """ConvTranspose (= conv data-gradient) with fp16 accumulation; precompiled plans only."""
    H16 = cudnn.data_type.HALF
    nd = w.dim() - 2
    h = _get_handle()
    try:
        g = cudnn.pygraph(io_data_type=H16, intermediate_data_type=H16, compute_data_type=H16, handle=h)
        DY = g.tensor(name="DY", dim=list(x.shape), stride=list(x.stride()), data_type=H16)
        W = g.tensor(name="W", dim=list(w.shape), stride=list(w.stride()), data_type=H16)
        DX = g.conv_dgrad(loss=DY, filter=W, padding=list(padding), stride=list(stride), dilation=[1] * nd,
                          compute_data_type=H16)
        DX.set_output(True).set_data_type(H16).set_dim(yshape).set_stride(_cl_stride(yshape))
        g.validate()
        g.build_operation_graph()
        g.deselect_behavior_notes([cudnn.behavior_note.RUNTIME_COMPILATION])
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
    except Exception:  # noqa: BLE001
        return None
    for i in range(g.get_execution_plan_count()):
        if g.get_plan_name_at_index(i).split("_")[0] in DENY:
            continue
        try:
            g.build_plan_at_index(i)
        except Exception:  # noqa: BLE001
            continue
        if VERIFY:
            xr = (torch.rand(x.shape, device=x.device) * 2 - 1).half().contiguous(memory_format=_mf(x.dim()))
            y = torch.empty(yshape, device=x.device, dtype=torch.half, memory_format=_mf(x.dim()))
            ws = torch.empty(max(g.get_workspace_size_plan_at_index(i), 1), device=x.device, dtype=torch.uint8)
            g.execute_plan_at_index({DY: xr, W: w, DX: y}, ws, i, handle=h)
            f = F.conv_transpose2d if x.dim() == 4 else F.conv_transpose3d
            ref = f(xr.float(), w.float(), None, stride, padding)
            dev = float((y.float() - ref).abs().max() / (ref.abs().max() + 1e-3))
            if not (dev == dev and dev < 0.05):
                print(f"[cudnn_conv] rejected dgrad {g.get_plan_name_at_index(i)}: {dev:.3g}")
                continue
        stats["built"] += 1
        return g, DY, W, DX, i, g.get_workspace_size_plan_at_index(i), yshape
    return None


def deconv_half_acc(x, w, padding, stride):
    """ConvTranspose (output_padding 0, groups 1, dilation 1) fp16 with fp16 accumulation, channels-last out."""
    mf = _mf(x.dim())
    x = x.contiguous(memory_format=mf)
    sp = [(x.shape[2 + i] - 1) * stride[i] - 2 * padding[i] + w.shape[2 + i] for i in range(w.dim() - 2)]
    yshape = [x.shape[0], w.shape[1], *sp]
    key = ("dgrad", tuple(x.shape), tuple(w.shape), tuple(padding), tuple(stride), x.device.index)
    e = _plans.get(key, False)
    if e is False:
        e = _plans[key] = _build_dgrad(x, w, padding, stride, yshape)
    if e is None:
        stats["torch"] += 1
        f = F.conv_transpose2d if x.dim() == 4 else F.conv_transpose3d
        return f(x, w, None, stride, padding).contiguous(memory_format=mf)
    stats["cudnn"] += 1
    g, DY, W, DX, i, ws_size, yshape = e
    y = torch.empty(yshape, device=x.device, dtype=x.dtype, memory_format=mf)
    ws = _ws[0]
    if ws is None or ws.numel() < ws_size:
        ws = _ws[0] = torch.empty(max(ws_size, 1), device=x.device, dtype=torch.uint8)
    h = _get_handle()
    cudnn.set_stream(handle=h, stream=torch.cuda.current_stream().cuda_stream)
    g.execute_plan_at_index({DY: x, W: w, DX: y}, ws, i, handle=h)
    return y


def _forward(self, x):
    return conv_half_acc(x, self.weight, self.bias, list(self.padding), list(self.stride))


def patch_convs(module, mark_only=False):
    """Route every fp16 nn.Conv2d / nn.Conv3d (groups 1, dilation 1, zero padding) of `module` through conv_half_acc.
    mark_only: only set m._hacc (the caller's forward uses conv_half_acc); output channels are then zero-padded to a
    multiple of 8 (tensor-core friendly; the caller drops the extra outputs)."""
    assert cudnn is not None, "pip install nvidia-cudnn-frontend"
    n = 0
    for m in module.modules():
        if mark_only and type(m) in (nn.ConvTranspose2d, nn.ConvTranspose3d) and m.groups == 1 \
                and all(d == 1 for d in m.dilation) and not any(m.output_padding) and m.weight.dtype == torch.half:
            m.weight.data = m.weight.data.contiguous(memory_format=_mf(m.weight.dim()))
            m._hacc_t = True
            n += 1
        if type(m) in (nn.Conv2d, nn.Conv3d) and m.groups == 1 and all(d == 1 for d in m.dilation) \
                and m.padding_mode == "zeros" and not isinstance(m.padding, str) and m.weight.dtype == torch.half:
            w = m.weight.data
            if mark_only and w.shape[0] % 8:
                pad = torch.zeros((8 - w.shape[0] % 8,) + tuple(w.shape[1:]), dtype=w.dtype, device=w.device)
                w = torch.cat([w, pad], 0)
                if getattr(m, "_bias32", None) is not None:
                    m._bias32 = torch.cat([m._bias32, m._bias32.new_zeros(pad.shape[0])]).contiguous()
            m.weight.data = w.contiguous(memory_format=_mf(w.dim()))
            if mark_only:
                m._hacc = True
            else:
                m.forward = _forward.__get__(m)
            n += 1
    return n
