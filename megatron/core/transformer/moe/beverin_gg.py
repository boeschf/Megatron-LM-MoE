# beverin-rocm (lab T4): Primus-Turbo's persistent Triton grouped GEMM (primus_turbo 0.2.0,
# triton/grouped_gemm/grouped_gemm_kernel.py, Copyright (c) 2025 Advanced Micro Devices, Inc.) with one change:
# the weight of group g is found through a per-group element-offset table instead of a [G, N, K] stack.
# Megatron's DDP buffer does not keep the 16 local expert weights of a layer back to back, so the
# stack view fails and every expert GEMM call (fwd, recompute fwd, dgrad) copied all weights with
# torch.stack (about 0.3 s per step on the last stage).
import torch
import triton
import triton.language as tl
from primus_turbo.triton.gemm.gemm_kernel import _is_gfx950, _set_knobs_gfx950
from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
    NUM_XCDS, _chiplet_transform_chunked, _get_gg_bf16_fwd_config, _get_num_cus,
)


@triton.jit()
def _beverin_gg_ptr_kernel(
    # Pointers
    A,  # [M_total, K]
    B,  # [G, ?, ?]  — (K,N) or (N,K) depending on trans_b
    C,  # [M_total, N]
    group_offs_ptr,  # [G+1] int64
    b_goff_ptr,  # [G] int64: element offset of group g's weight from B (weights need not be stacked)
    # Dimensions
    G,  # number of groups (runtime)
    N,
    K,
    # Strides
    stride_am,  # A row stride
    stride_bg,  # B group stride: b.stride(0)
    stride_bn,  # B N-stride (within a group)
    stride_cm,  # C row stride
    stride_cn,  # C col stride
    # Constexpr strides (for compiler optimisation)
    stride_ak: tl.constexpr,  # A K-stride (=1 when trans_a=False, contiguous)
    stride_bk: tl.constexpr,  # B K-stride (=1 when trans_b=True)
    # Tile config
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    CACHE_MODIFIER_A: tl.constexpr,
    CACHE_MODIFIER_B: tl.constexpr,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    """Persistent grouped GEMM kernel (CPU-sync-free).

    One kernel launch processes ALL groups × ALL tiles.
    Each persistent CU computes total_tiles and maps global tile IDs to
    (group, local_tile) on the fly via O(G) linear scan of group_offs.
    """
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = _chiplet_transform_chunked(pid, NUM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # ── Compute total tiles across all groups (O(G) per CU, group_offs cached in L2) ──
    # Cast int64 group_offs to int32 for tile arithmetic (tile counts fit in int32)
    total_tiles: tl.int32 = 0
    for _g in range(G):
        m_g = (tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)).to(tl.int32)
        total_tiles += tl.cdiv(m_g, BLOCK_SIZE_M) * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32

    for global_tile_id in range(pid, total_tiles, NUM_SMS):
        # ── Find group via linear scan (O(G)) ──
        group_idx: tl.int32 = 0
        tile_start: tl.int32 = 0
        cumsum: tl.int32 = 0
        for _g in range(G):
            m_g_i = (tl.load(group_offs_ptr + _g + 1) - tl.load(group_offs_ptr + _g)).to(tl.int32)
            tiles_g = tl.cdiv(m_g_i, BLOCK_SIZE_M) * num_pid_n
            new_cumsum = cumsum + tiles_g
            if global_tile_id >= new_cumsum:
                group_idx = _g + 1
                tile_start = new_cumsum
            cumsum = new_cumsum

        # ── Group-local tile → (pid_m, pid_n) with GROUP_SIZE_M swizzle ──
        local_tile = global_tile_id - tile_start
        m_start_g = tl.load(group_offs_ptr + group_idx)  # keep int64 to avoid address overflow
        M_g = (tl.load(group_offs_ptr + group_idx + 1) - tl.load(group_offs_ptr + group_idx)).to(tl.int32)
        tiles_m_g = tl.cdiv(M_g, BLOCK_SIZE_M)

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        swizzle_group = local_tile // num_pid_in_group
        first_pid_m = swizzle_group * GROUP_SIZE_M
        group_size_m = min(tiles_m_g - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((local_tile % num_pid_in_group) % group_size_m)
        pid_n = (local_tile % num_pid_in_group) // group_size_m
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        # ── Address computation ──
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Cast group_idx to int64 to prevent overflow in B group offset
        # (group_idx * stride_bg can exceed int32 when B has many groups)
        group_offset_b = tl.multiple_of(tl.load(b_goff_ptr + group_idx), 16)
        tl.assume(group_offset_b >= 0)

        A_BASE = A + m_start_g * stride_am + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + group_offset_b + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        # ── K-loop (identical to single GEMM) ──
        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        tl.assume(loop_k > 1)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_A)
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_A)

            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)), cache_modifier=CACHE_MODIFIER_B)
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)), cache_modifier=CACHE_MODIFIER_B)

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            rk_last = loop_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_LAST = A + m_start_g * stride_am + rm[:, None] * stride_am + rk_last[None, :] * stride_ak
            B_LAST = B + group_offset_b + rk_last[:, None] * stride_bk + rn[None, :] * stride_bn
            if stride_ak == 1:
                A_LAST = tl.multiple_of(A_LAST, (1, 16))
            else:
                A_LAST = tl.multiple_of(A_LAST, (16, 1))
            if stride_bk == 1:
                B_LAST = tl.multiple_of(B_LAST, (16, 1))
            else:
                B_LAST = tl.multiple_of(B_LAST, (1, 16))
            a = tl.load(A_LAST, mask=rk_last[None, :] < K, other=0.0, cache_modifier=CACHE_MODIFIER_A)
            b = tl.load(B_LAST, mask=rk_last[:, None] < K, other=0.0, cache_modifier=CACHE_MODIFIER_B)
            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

        # ── Store ──
        c = acc.to(C.type.element_ty)
        rm_s = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M_g
        rn_s = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rn_s = tl.max_contiguous(tl.multiple_of(rn_s, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm_s[:, None] < M_g) & (rn_s[None, :] < N)
        C_ = C + m_start_g * stride_cm + rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn
        tl.store(C_, c, c_mask)



def grouped_gemm_ptr(
    a: torch.Tensor,
    b: torch.Tensor,
    group_offs: torch.Tensor,
    b_goff: torch.Tensor,
    trans_b: bool = False,
) -> torch.Tensor:
    """As Primus-Turbo grouped_gemm_triton_kernel, but b is only group 0's weight viewed as [1, ., .] and
    b_goff[g] (int64, elements) locates group g's weight relative to it."""
    """Persistent grouped GEMM (CPU-sync-free) using Triton.

    Computes: out[offs[g]:offs[g+1], :] = a[offs[g]:offs[g+1], :] @ B_view[g]
    for g = 0, ..., G-1, where B_view[g] is b[g] or b[g]^T depending on trans_b.

    Single kernel launch, zero CPU synchronization.

    Args:
        a: [M_total, K] BF16/FP16 input (trans_a=False always).
        b: [G, K, N] or [G, N, K] (if trans_b) BF16/FP16 weights.
        group_offs: [G+1] int64 prefix sum of group lengths.
        trans_b: If True, b[g] is [N, K] (transposed).

    Returns:
        [M_total, N] BF16/FP16 output.
    """
    assert a.ndim == 2, f"a must be 2D, got {a.shape}"
    assert b.ndim == 3, f"b must be 3D, got {b.shape}"
    assert a.dtype in (torch.bfloat16, torch.float16), f"Unsupported dtype: {a.dtype}"
    assert b.dtype in (torch.bfloat16, torch.float16), f"Unsupported dtype: {b.dtype}"

    M_total, K_a = a.shape
    G = b_goff.numel()

    if trans_b:
        N, K_b = b.shape[1], b.shape[2]
        stride_bk = b.stride(2)  # K is the fast dimension (=1 for contiguous)
        stride_bn = b.stride(1)  # N-stride
    else:
        K_b, N = b.shape[1], b.shape[2]
        stride_bk = b.stride(1)  # K-stride
        stride_bn = b.stride(2)  # N is the fast dimension (=1 for contiguous)

    assert K_a == K_b, f"K mismatch: a has K={K_a}, b has K={K_b}"
    K = K_a

    stride_bg = b.stride(0)  # Group stride
    stride_ak = a.stride(1)  # =1 for contiguous a

    # Output
    out = torch.empty((M_total, N), device=a.device, dtype=a.dtype)

    # Kernel config (cached — origami + LDS check run only on first call per shape)
    num_sms = _get_num_cus()
    avg_m = max(M_total // max(G, 1), 256)
    if _is_gfx950():
        _set_knobs_gfx950()
    BLOCK_M, BLOCK_N, BLOCK_K, group_m, cache_a, cache_b, num_stages_val, chunk_size = (
        _get_gg_bf16_fwd_config(avg_m, N, K, a.dtype, b.dtype, trans_b, G, num_sms)
    )
    even_k = K % BLOCK_K == 0

    _beverin_gg_ptr_kernel[(num_sms,)](
        a,
        b,
        out,
        group_offs,
        b_goff,
        G,
        N,
        K,
        a.stride(0),  # stride_am
        stride_bg,  # B group stride
        stride_bn,  # B N-stride
        out.stride(0),  # stride_cm
        out.stride(1),  # stride_cn
        stride_ak=stride_ak,
        stride_bk=stride_bk,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=group_m,
        NUM_SMS=num_sms,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=chunk_size,
        EVEN_K=even_k,
        CACHE_MODIFIER_A=cache_a,
        CACHE_MODIFIER_B=cache_b,
        num_warps=8,
        num_stages=num_stages_val,
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return out


