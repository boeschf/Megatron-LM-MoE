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
# K6: load-balanced causal forward at batch 1 (see _k6_chunked). BEVERIN_K6_CHUNKS lists the q-tile chunks
# in launch order as "s-e" tile ranges (tiles of 256 rows), or "auto" for the tuned default.
_K6 = os.environ.get("BEVERIN_K6_CHUNKS", "")
_K6_PLANS = {}


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


def _k6_plan(S, device):
    """Chunks (s, e) of 256-row q tiles in launch order (roughly heaviest first). Default at seq 8192: the
    plan found by K6's search on MI300A; other lengths: single tiles from the end, first 4 tiles as pairs."""
    key = (S, str(device))
    if key not in _K6_PLANS:
        n = S // 256
        if _K6 in ("1", "auto") and n == 32:
            # K6 empirical search (job 681793, search.py): 0.862 ms vs stock 1.07 at mid-4b MBS 1
            plan = [(22,23), (30,31), (19,20), (25,26), (27,28), (20,22), (31,32), (24,25), (23,24), (28,29), (26,27), (29,30), (11,12), (17,18), (15,16), (10,11), (14,15), (13,14), (12,13), (18,19), (16,17), (9,10), (8,9), (7,8), (6,7), (5,6), (0,1), (3,4), (1,3), (4,5)]
        elif _K6 in ("1", "auto"):
            plan = [(i, i + 1) for i in range(n - 1, 3, -1)] + [(2, 4), (0, 2)]
        else:
            plan = [tuple(int(x) for x in c.split("-")) for c in _K6.split(",")]
        assert sorted(t for s, e in plan for t in range(s, e)) == list(range(n)), "K6 chunks must tile the sequence"
        i32 = lambda x: torch.tensor(x, device=device, dtype=torch.int32)
        lq = [(e - s) * 256 for s, e in plan]
        lk = [e * 256 for s, e in plan]
        cu_q = i32([0] + torch.tensor(lq).cumsum(0).tolist())
        cu_k = i32([0] + torch.tensor(lk).cumsum(0).tolist())
        pq = i32([s * 256 for s, _ in plan] + [S])
        pk = i32([0] * len(plan) + [S])
        _K6_PLANS[key] = (cu_q, cu_k, pq, pk, max(lq))
    return _K6_PLANS[key]


class _K6ChunkedCausal(torch.autograd.Function):
    """Causal attention, batch 1. Forward: one launch of aiter's asm causal group-mode kernel over q-row
    chunks, each chunk against k[0:end] with a bottom-right causal mask, heaviest chunk first, so the
    hardware dispatcher balances the 228 CUs (the stock launch is 320 equal workgroups: 1.4 rounds).
    Output and lse equal the stock forward's; the backward is aiter's stock asm causal backward."""

    @staticmethod
    def forward(ctx, q, k, v, scale, cvt):
        from aiter.ops.mha import fmha_v3_varlen_fwd

        _, S, H, D = q.shape
        cu_q, cu_k, pq, pk, mq = _k6_plan(S, q.device)
        q2, k2, v2 = q.reshape(S, H, D), k.reshape(S, k.shape[2], D), v.reshape(S, v.shape[2], D)
        r = fmha_v3_varlen_fwd(q2, k2, v2, cu_q, cu_k, mq, mq, 0, 0.0, scale, 0.0, False, True, -1, 0, True, False,
                               cvt, None, None, None, None, None, None, None, None, pq, pk)
        out = r[0].view(1, S, H, D)
        lse = r[1].view(1, H, S)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale, ctx.cvt = scale, cvt
        return out

    @staticmethod
    def backward(ctx, do):
        from aiter.ops.mha import _flash_attn_backward

        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        rng = torch.empty(2, device=q.device, dtype=torch.int64)
        _flash_attn_backward(do.contiguous(), q, k, v, out, lse, dq, dk, dv, None, 0.0, ctx.scale, True, -1, -1,
                             None, None, False, rng, True, ctx.cvt)
        return dq, dk, dv, None, None


def forward(module, query, key, value, packed_seq_params=None):
    import aiter

    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"[beverin] aiter flash attention for causal TE attention (cvt={_CVT}, k6 chunks={_K6 or 'off'})", flush=True)
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
    if _K6 and b == 1 and s % 256 == 0 and s >= 4096:
        q, k, v = (t.transpose(0, 1).contiguous() for t in (query, key, value))  # views at b == 1
        out = _K6ChunkedCausal.apply(q, k, v, scale, _CVT)
        return out.transpose(0, 1).reshape(s, b, -1)
    out = aiter.flash_attn_func(
        query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1), causal=True,
        softmax_scale=scale, deterministic=False, return_lse=True, how_v3_bf16_cvt=_CVT)
    out = out[0] if isinstance(out, (tuple, list)) else out
    return out.transpose(0, 1).reshape(s, b, -1)
