import os
# beverin-rocm (lab T4): Primus-Turbo's persistent Triton variable-K grouped GEMM (expert weight gradient; primus_turbo
# 0.2.0 triton/grouped_gemm/grouped_gemm_kernel.py, Copyright (c) 2025 Advanced Micro Devices, Inc.) changed to
# accumulate C[g] = grad_out_g^T @ x_g directly into each expert's fp32 main_grad (found through a per-group element
# offset table; the experts' main_grads are scattered in the DDP buffer). Replaces bf16 wgrad output + 2 x 16
# separate main_grad.add_ per MoE layer and microbatch, and accumulates in fp32 without a bf16 rounding of each
# microbatch's gradient (as TE's gradient-accumulation fusion does on GH200).
import torch
import triton
import triton.language as tl
from primus_turbo.triton.gemm.gemm_kernel import _is_gfx950, _set_knobs_gfx950
from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
    NUM_XCDS, _chiplet_transform_chunked, _get_gg_bf16_vk_config, _get_num_cus,
)


@triton.jit()
def _beverin_vk_accum_kernel(
    # C[g] = LHS_g^T @ RHS_g [* scale if IS_FP8]
    # LHS: [M_total, OUT_M] (2D), RHS: [M_total, OUT_N] (2D)
    # C: [G, OUT_M, OUT_N] (3D)
    LHS,
    RHS,
    C,
    LHS_scale_ptr,
    RHS_scale_ptr,  # only used if IS_FP8
    group_offs_ptr,  # [G+1] int64
    c_goff_ptr,  # [G] int64: element offset of group g's fp32 output (main_grad) from C
    G,  # number of groups
    OUT_M,
    OUT_N,  # output dimensions (fixed across groups)
    # Strides
    stride_lhs_m,  # LHS row stride (along M_total)
    stride_rhs_m,  # RHS row stride (along M_total)
    stride_cg,  # C group stride
    stride_cm,  # C row stride (along OUT_M)
    stride_cn,  # C col stride (along OUT_N)
    # Constexpr strides
    stride_lhs_n: tl.constexpr,  # LHS col stride (=1 for row-major)
    stride_rhs_n: tl.constexpr,  # RHS col stride (=1 for row-major)
    # Tile config
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # inner loop block over M_g
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    IS_FP8: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
    PTAB: tl.constexpr = False,  # bench2 K4: c_goff_ptr holds int64 addresses of each group's fp32 main grad
):
    """Persistent grouped variable-K GEMM kernel for backward pass (CPU-sync-free).

    All groups share the same output dims (OUT_M × OUT_N), only the inner product
    dimension M_g varies per group. Group→tile mapping is simple div/mod.
    """
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    tiles_m = tl.cdiv(OUT_M, BLOCK_SIZE_M)
    tiles_n = tl.cdiv(OUT_N, BLOCK_SIZE_N)
    tiles_per_group = tiles_m * tiles_n
    total_tiles = G * tiles_per_group

    tl.assume(stride_lhs_m > 0)
    tl.assume(stride_lhs_n > 0)
    tl.assume(stride_rhs_m > 0)
    tl.assume(stride_rhs_n > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    if IS_FP8:
        scale = tl.load(LHS_scale_ptr) * tl.load(RHS_scale_ptr)

    acc_dtype = tl.float32

    for global_tile in range(pid, total_tiles, NUM_SMS):
        # ── Map to (group, local_tile) — simple div/mod ──
        group_idx = global_tile // tiles_per_group
        local_tile = global_tile - group_idx * tiles_per_group

        # ── Swizzle local tile → (pid_m, pid_n) ──
        num_pid_in_group = GROUP_SIZE_M * tiles_n
        swizzle_group = local_tile // num_pid_in_group
        first_pid_m = swizzle_group * GROUP_SIZE_M
        group_size_m = min(tiles_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
        pid_n = (local_tile % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # ── Group boundaries ──
        m_start = tl.load(group_offs_ptr + group_idx)  # int64 to avoid overflow
        M_g = (tl.load(group_offs_ptr + group_idx + 1) - m_start).to(tl.int32)

        # ── Output indices ──
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % OUT_M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % OUT_N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # ── Base pointers ──
        # LHS^T[rm, rk] = LHS[m_start + rk, rm]
        LHS_BASE = LHS + m_start * stride_lhs_m + rm[:, None] * stride_lhs_n + rk[None, :] * stride_lhs_m
        # RHS[rk, rn] = RHS[m_start + rk, rn]
        RHS_BASE = RHS + m_start * stride_rhs_m + rk[:, None] * stride_rhs_m + rn[None, :] * stride_rhs_n

        # ── K-loop over M_g (variable per group, always masked for correctness) ──
        loop_k = tl.cdiv(M_g, BLOCK_SIZE_K)
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k in range(loop_k):
            k_start = k * BLOCK_SIZE_K
            mask_k = (k_start + tl.arange(0, BLOCK_SIZE_K)) < M_g

            if stride_lhs_n == 1:
                a = tl.load(
                    tl.multiple_of(LHS_BASE, (16, 1)),
                    mask=mask_k[None, :],
                    other=0.0,
                    cache_modifier=CACHE_MODIFIER_A,
                )
            else:
                a = tl.load(
                    tl.multiple_of(LHS_BASE, (1, 16)),
                    mask=mask_k[None, :],
                    other=0.0,
                    cache_modifier=CACHE_MODIFIER_A,
                )

            if stride_rhs_n == 1:
                b = tl.load(
                    tl.multiple_of(RHS_BASE, (1, 16)),
                    mask=mask_k[:, None],
                    other=0.0,
                    cache_modifier=CACHE_MODIFIER_B,
                )
            else:
                b = tl.load(
                    tl.multiple_of(RHS_BASE, (16, 1)),
                    mask=mask_k[:, None],
                    other=0.0,
                    cache_modifier=CACHE_MODIFIER_B,
                )

            if IS_FP8:
                acc += tl.dot(a, b, input_precision="ieee")
            else:
                acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

            LHS_BASE += BLOCK_SIZE_K * stride_lhs_m
            RHS_BASE += BLOCK_SIZE_K * stride_rhs_m

        # ── Apply scaling and store ──
        if IS_FP8:
            acc *= scale
        # beverin: accumulate into the fp32 main_grad of expert g (C + c_goff[g]), no bf16 round trip
        rm_s = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn_s = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s % OUT_N, BLOCK_SIZE_N), BLOCK_SIZE_N)
        # mask on the unwrapped column index: wrapped lanes would add the same tile twice
        rn_raw = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < OUT_M) & (rn_raw[None, :] < OUT_N)
        # Cast group_idx to int64 to prevent overflow in C group offset
        if PTAB:
            C_G = tl.load(c_goff_ptr + group_idx).to(tl.pointer_type(tl.float32))
        else:
            c_off = tl.multiple_of(tl.load(c_goff_ptr + group_idx), 16)
            tl.assume(c_off >= 0)
            C_G = C + c_off
        C_ = C_G + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        c_old = tl.load(C_, mask=c_mask, other=0.0)
        tl.store(C_, c_old + acc, c_mask)




def vk_accum(lhs, rhs, group_offs, c0, c_goff, ptab=False):
    """c0 + c_goff[g] (fp32, row-major [OUT_M, OUT_N]) += lhs_g^T @ rhs_g for every group g.
    lhs: [M_total, OUT_M] (grad_out), rhs: [M_total, OUT_N] (x), bf16, contiguous."""
    assert lhs.ndim == 2 and rhs.ndim == 2 and lhs.shape[0] == rhs.shape[0]
    assert c0.dtype == torch.float32 and c0.is_contiguous() and tuple(c0.shape) == (lhs.shape[1], rhs.shape[1])
    OUT_M, OUT_N = lhs.shape[1], rhs.shape[1]
    G = group_offs.shape[0] - 1
    num_sms = _get_num_cus()
    _cap = os.environ.get("BEVERIN_TURBO_NUM_CU")  # lab2 A3: persistent grid on fewer CUs
    if _cap:
        num_sms = max(NUM_XCDS, min(num_sms, int(_cap)) // NUM_XCDS * NUM_XCDS)
    _o1cap = os.environ.get("BEVERIN_O1_VK_CU")  # bench2 O1: fused wgrad grid on fewer CUs (leave CUs to RCCL)
    if _o1cap:
        num_sms = max(NUM_XCDS, min(num_sms, int(_o1cap)) // NUM_XCDS * NUM_XCDS)
    dummy = torch.empty(1, device=lhs.device, dtype=torch.float32)
    if _is_gfx950():
        _set_knobs_gfx950()
    avg_m_g = max(lhs.shape[0] // max(G, 1), 256)
    BLOCK_M, BLOCK_N, BLOCK_K, group_m, cache_a, cache_b, num_stages_val, chunk_size = _get_gg_bf16_vk_config(
        OUT_M, OUT_N, avg_m_g, lhs.dtype, rhs.dtype, G, num_sms)
    _beverin_vk_accum_kernel[(num_sms,)](
        lhs, rhs, c0, dummy, dummy, group_offs, c_goff, G, OUT_M, OUT_N,
        lhs.stride(0), rhs.stride(0), OUT_M * OUT_N, c0.stride(0), c0.stride(1),
        stride_lhs_n=lhs.stride(1), stride_rhs_n=rhs.stride(1),
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K, GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms, NUM_XCDS=NUM_XCDS, CHUNK_SIZE=chunk_size, IS_FP8=False,
        CACHE_MODIFIER_A=cache_a, CACHE_MODIFIER_B=cache_b,
        num_warps=8, num_stages=num_stages_val, waves_per_eu=0, matrix_instr_nonkdim=16, kpack=1, PTAB=ptab,
    )
