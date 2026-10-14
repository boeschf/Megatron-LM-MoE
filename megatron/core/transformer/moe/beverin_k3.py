"""K3 (bench2) small Triton kernels for the router and the T6 plan (MI300A and GH200).

row_topk(x, k): torch.topk(x, k, dim=1) (sorted, largest) for 2D x with up to 1024 columns; ties go to the lowest index.
mask_to_idx(mask, k): [T, k] int64 column indices of the True entries of each row, ascending (the T6 plan's
    torch.topk(routing_map.to(torch.int8), k, sorted=False).indices, which only needs the set). Rows with fewer
    than k True entries are padded with the lowest False columns (as topk over int8 would pick zeros).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _row_topk_kernel(x_ptr, v_ptr, i_ptr, T, N, stride_x, K: tl.constexpr, BR: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    r = (pid * BR + tl.arange(0, BR)).to(tl.int64)
    c = tl.arange(0, BN)
    rm = r < T
    m = rm[:, None] & (c[None, :] < N)
    x = tl.load(x_ptr + r[:, None] * stride_x + c[None, :], mask=m, other=float("-inf")).to(tl.float32)
    for j in tl.static_range(K):
        mx = tl.max(x, axis=1)
        hit = x == mx[:, None]
        ix = tl.min(tl.where(hit, c[None, :], BN), axis=1)
        tl.store(v_ptr + r * K + j, mx.to(v_ptr.dtype.element_ty), mask=rm)
        tl.store(i_ptr + r * K + j, ix.to(tl.int64), mask=rm)
        x = tl.where(c[None, :] == ix[:, None], float("-inf"), x)


def row_topk(x, k):
    assert x.ndim == 2 and x.shape[1] <= 1024
    T, N = x.shape
    if x.stride(1) != 1:
        x = x.contiguous()
    v = torch.empty(T, k, dtype=x.dtype, device=x.device)
    i = torch.empty(T, k, dtype=torch.int64, device=x.device)
    BN = triton.next_power_of_2(N)
    BR = max(1, 2048 // BN)
    _row_topk_kernel[(triton.cdiv(T, BR),)](x, v, i, T, N, x.stride(0), K=k, BR=BR, BN=BN, num_warps=4)
    return v, i


@triton.jit
def _mask_to_idx_kernel(m_ptr, o_ptr, T, N, stride_m, K: tl.constexpr, BR: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    r = (pid * BR + tl.arange(0, BR)).to(tl.int64)
    c = tl.arange(0, BN)
    rm = r < T
    valid = rm[:, None] & (c[None, :] < N)
    v = tl.load(m_ptr + r[:, None] * stride_m + c[None, :], mask=valid, other=0).to(tl.int32)
    v = tl.where(valid, (v != 0).to(tl.int32), 0)
    pos = tl.cumsum(v, axis=1) - 1  # slot of each True entry
    cnt = tl.sum(v, axis=1)
    nf = tl.cumsum(1 - v, axis=1) - 1  # rank of each False entry among the False ones
    slot = tl.where(v == 1, pos, cnt[:, None] + nf)
    ok = valid & (slot < K)
    tl.store(o_ptr + r[:, None] * K + slot, c[None, :].to(tl.int64) + tl.zeros([BR, BN], dtype=tl.int64), mask=ok)


def mask_to_idx(mask, k):
    assert mask.ndim == 2 and mask.shape[1] <= 1024
    T, N = mask.shape
    if mask.stride(1) != 1:
        mask = mask.contiguous()
    m8 = mask.view(torch.uint8) if mask.dtype == torch.bool else mask
    o = torch.empty(T, k, dtype=torch.int64, device=mask.device)
    BN = triton.next_power_of_2(N)
    BR = max(1, 2048 // BN)
    _mask_to_idx_kernel[(triton.cdiv(T, BR),)](m8, o, T, N, m8.stride(0), K=k, BR=BR, BN=BN, num_warps=4)
    return o
