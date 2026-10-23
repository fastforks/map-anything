"""Vendored copy of parts of fastforks/fastbb (own copy, no dependency on the fastbb package). Do not edit here:
change fastbb and re-sync.
files @ last-changed fastbb commit (a61ba9c): attention.py cudnn_conv.py fastbb_kernels.cu kernels.py vit.py
"""


def parse_vit_spec(spec):
    attn, _, g = spec.partition("-")
    gacc = {"g0": False, "g1": True, "g2": "nofc2", "": False}[g]  # g3 (chunked GEMM kernel) is not vendored
    return attn, gacc
