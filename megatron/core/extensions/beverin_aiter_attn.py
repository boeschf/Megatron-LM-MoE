# Beverin (bench2 B3): global causal attention through aiter's flash-attention API on ROCm.
#
# Why: on the primus_v26.3 image, TE's DotProductAttention either picks flash_attn 2.8.3
# (CK tile kernels, about 1.9x slower than the asm path at head dim 128) or, with
# NVTE_FLASH_ATTN=0, TE's fused CK path whose v3 (asm) backward returns wrong dq and dk
# (relative error 0.4 to 0.55 against an fp32 reference; dv and the forward are right).
# aiter.flash_attn_func with return_lse=True, deterministic=False runs the same asm
# forward and backward kernels with correct gradients (2.4e-3 relative error).
#
# Enabled with BEVERIN_AITER_ATTN=1 (default off). BEVERIN_AITER_ATTN_CVT selects the
# bf16 rounding of the asm kernels (0 RTNE, 1 RTNA (default), 2 RTZ: faster, about 2.5x
# larger output error).
import os

import torch

_ON = os.environ.get("BEVERIN_AITER_ATTN", "0") == "1"
_CVT = int(os.environ.get("BEVERIN_AITER_ATTN_CVT", "1"))
_LOGGED = False


def enabled() -> bool:
    return _ON


def usable(module, attention_mask, attn_mask_type, attention_bias, packed_seq_params) -> bool:
    """Only the plain causal case without bias, dropout, CP, qk-clip or softmax offsets."""
    if not _ON:
        return False
    from megatron.core.transformer.enums import AttnMaskType

    cfg = module.config
    if attention_bias is not None or attn_mask_type not in (AttnMaskType.causal, AttnMaskType.padding_causal):
        return False
    if getattr(cfg, "qk_clip", False) or getattr(cfg, "log_max_attention_logit", False):
        return False
    if getattr(cfg, "softmax_type", "vanilla") != "vanilla" or cfg.window_size is not None:
        return False
    if module.training and cfg.attention_dropout > 0:
        return False
    if getattr(cfg, "context_parallel_size", 1) > 1:
        return False
    if packed_seq_params is not None and packed_seq_params.qkv_format != "thd":
        return False
    return True


def forward(module, query, key, value, packed_seq_params=None):
    import aiter

    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"[beverin] aiter flash attention for causal TE attention (cvt={_CVT})", flush=True)
    scale = getattr(module, "softmax_scale", None)
    if scale is None:
        scale = module.fused_attention.softmax_scale
    if packed_seq_params is not None:
        p = packed_seq_params
        cu_q = p.cu_seqlens_q_padded if p.cu_seqlens_q_padded is not None else p.cu_seqlens_q
        cu_k = p.cu_seqlens_kv_padded if p.cu_seqlens_kv_padded is not None else p.cu_seqlens_kv
        # thd: [t, h, d] (Megatron may pass [t, 1, h, d])
        squeeze = query.dim() == 4
        q, k, v = (t.squeeze(1) if squeeze else t for t in (query, key, value))
        out = aiter.flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, p.max_seqlen_q, p.max_seqlen_kv, softmax_scale=scale,
            causal=True, deterministic=False, return_lse=True, how_v3_bf16_cvt=_CVT)
        out = out[0] if isinstance(out, (tuple, list)) else out
        return out.reshape(out.shape[0], -1)
    s, b = query.shape[0], query.shape[1]
    out = aiter.flash_attn_func(
        query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), causal=True,
        softmax_scale=scale, deterministic=False, return_lse=True, how_v3_bf16_cvt=_CVT)
    out = out[0] if isinstance(out, (tuple, list)) else out
    return out.transpose(0, 1).reshape(s, b, -1)
