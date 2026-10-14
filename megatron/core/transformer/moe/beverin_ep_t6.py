# beverin-rocm (lab T6): hierarchical, rail-aligned, deduplicated MoE dispatch/combine for EP groups that span
# several nodes (4 MI300A per node). Successor of T3's beverin_hier_a2a.py (same routing semantics), with:
#   - stage 1 (inter-node) on a 4-rank "rail" communicator (EP ranks with the same local rank, one per node), so every
#     GPU talks only to its rail peers over its own NIC; stage 2 (intra-node, xGMI) on a 4-rank node communicator.
#     Both are created and connected when the object is built (no lazy CXI endpoint creation late in training).
#   - one packed byte buffer per stage: [x (bf16, or fp8 + one fp32 scale per 128 columns) | probs fp32 K | ids int16 K]
#   - optional fp8 dispatch (BEVERIN_T6_FP8=1): the forward dispatch carries fp8 rows through both stages and
#     dequantises once, in the final expert-sorted gather; gradients and the combine stay bf16 (as DeepEP on GH200).
#   - dispatch and combine are each one autograd.Function whose backward is the mirror path (no autograd graph over
#     the intermediate copies).
#   - optional chunking of the dispatch/combine (BEVERIN_T6_CHUNKS=n) so stage 2 of chunk i overlaps stage 1 of
#     chunk i+1 (different communicators, separate side streams).
# Reductions (copies of one row on the way back) are deterministic gather-sums in fp32 (no atomics).
import os

import torch
import torch.distributed as dist

_FP8 = os.environ.get("BEVERIN_T6_FP8", "0") == "1"
_TIMING = os.environ.get("BEVERIN_T6_TIMING", "0") == "1"
# BEVERIN_T6_DEVTPE=1 (T6 t6-0005, ported by A4): hand the experts a device copy of tokens-per-expert (from the plan's
# all-gather, already on the device), so the turbo expert MLP does no blocking H2D copy (forward and expert recompute).
_DEVTPE = os.environ.get("BEVERIN_T6_DEVTPE", "0") == "1"
_K3_TOPK = os.environ.get("BEVERIN_K3_TOPK", "0") == "1"
_K3_GSUM = os.environ.get("BEVERIN_K3_GSUM", "0") == "1"
_QB = 128  # fp8 block: columns per scale
_STASH = os.environ.get("BEVERIN_DISPATCH_STASH", "0") == "1"
# C1 (bench2): fused pack / gather-sum kernels, int32 stable sort, count GEMV in the plan (default off)
_FUSED = os.environ.get("BEVERIN_T6_FUSED", "0") == "1"  # lab2 A6: keep r2 for the expert recompute
TIMES = {}


class _tm:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        if _TIMING:
            self.s = torch.cuda.Event(enable_timing=True)
            self.s.record()

    def __exit__(self, *a):
        if _TIMING:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            TIMES.setdefault(self.label, []).append((self.s, e))


def timing_summary(reset=True):
    torch.cuda.synchronize()
    out = {k: sorted(s.elapsed_time(e) for s, e in v)[len(v) // 2] for k, v in TIMES.items()}  # median
    if reset:
        TIMES.clear()
    return out


def _fp8_dtype():
    arch = getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "gcnArchName", "")
    return torch.float8_e4m3fnuz if "gfx94" in arch else torch.float8_e4m3fn


# ------------------------------------------------------------------------------------------------------------------
# kernels
try:
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["n_src"])
    def _gather_sum_kernel(src_ptr, inv_ptr, out_ptr, n_src, H, src_stride, out_stride, W: tl.constexpr,
                           BLOCK: tl.constexpr):
        r = tl.program_id(0)
        hb = tl.program_id(1)
        cols = hb * BLOCK + tl.arange(0, BLOCK)
        m = cols < H
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for w in tl.static_range(W):
            j = tl.load(inv_ptr + r * W + w)
            if j < n_src:
                acc += tl.load(src_ptr + j.to(tl.int64) * src_stride + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(out_ptr + r.to(tl.int64) * out_stride + cols, acc.to(out_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _quant_gather_kernel(x_ptr, idx_ptr, q_ptr, s_ptr, x_stride, row_bytes, s_off, H, NB,
                             NBP: tl.constexpr, QB: tl.constexpr, FMAX: tl.constexpr):
        # q row r (bytes [0, H)) = fp8(x[idx[r]] / s), s (fp32 at byte s_off of the row) per QB columns
        r = tl.program_id(0)
        src = tl.load(idx_ptr + r).to(tl.int64)
        b = tl.arange(0, NBP)
        c = tl.arange(0, QB)
        bm = b < NB
        cols = b[:, None] * QB + c[None, :]
        m = bm[:, None] & (c[None, :] < QB)
        x = tl.load(x_ptr + src * x_stride + cols, mask=m, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=1)
        s = tl.maximum(amax, 1e-20) / FMAX
        q = (x / s[:, None]).to(tl.float8e4b8)
        tl.store(q_ptr + r.to(tl.int64) * row_bytes + cols, q.to(tl.uint8, bitcast=True), mask=m)
        sp = (s_ptr + r.to(tl.int64) * row_bytes + s_off).to(tl.pointer_type(tl.float32))
        tl.store(sp + b, s, mask=bm)

    @triton.jit
    def _dequant_gather_kernel(buf_ptr, rows_ptr, out_ptr, row_bytes, s_off, H, NB,
                               NBP: tl.constexpr, QB: tl.constexpr):
        r = tl.program_id(0)
        src = tl.load(rows_ptr + r).to(tl.int64)
        b = tl.arange(0, NBP)
        c = tl.arange(0, QB)
        bm = b < NB
        cols = b[:, None] * QB + c[None, :]
        m = bm[:, None] & (c[None, :] < QB)
        q = tl.load(buf_ptr + src * row_bytes + cols, mask=m, other=0).to(tl.float8e4b8, bitcast=True).to(tl.float32)
        sp = (buf_ptr + src * row_bytes + s_off).to(tl.pointer_type(tl.float32))
        s = tl.load(sp + b, mask=bm, other=0.0)
        tl.store(out_ptr + r.to(tl.int64) * H + cols, (q * s[:, None]).to(out_ptr.dtype.element_ty), mask=m)

    # ---- C1 (bench2): fused T6 front and back ends (BEVERIN_T6_FUSED=1) ----
    @triton.jit(do_not_specialize=["x_stride", "p_stride"])
    def _pack1_kernel(x_ptr, idx1_ptr, eid_ptr, probs_ptr, buf_ptr, x_stride, p_stride, row_bytes, s_off, po, io,
                      H, NB, K: tl.constexpr, NBP: tl.constexpr, QB: tl.constexpr, FMAX: tl.constexpr,
                      FP8: tl.constexpr, HB: tl.constexpr):
        # packed dispatch row r for token src = idx1[r]: x (fp8 + scales, or bf16) | probs[src, eid[src, :]] fp32 | eid int16
        r = tl.program_id(0)
        src = tl.load(idx1_ptr + r).to(tl.int64)
        row = buf_ptr + r.to(tl.int64) * row_bytes
        if FP8:
            b = tl.arange(0, NBP)
            c = tl.arange(0, QB)
            bm = b < NB
            cols = b[:, None] * QB + c[None, :]
            m = bm[:, None] & (c[None, :] < QB)
            x = tl.load(x_ptr + src * x_stride + cols, mask=m, other=0.0).to(tl.float32)
            amax = tl.max(tl.abs(x), axis=1)
            s = tl.maximum(amax, 1e-20) / FMAX
            q = (x / s[:, None]).to(tl.float8e4b8)
            tl.store(row + cols, q.to(tl.uint8, bitcast=True), mask=m)
            tl.store((row + s_off).to(tl.pointer_type(tl.float32)) + b, s, mask=bm)
        else:
            cc = tl.arange(0, HB)
            xm = cc < H
            xv = tl.load(x_ptr + src * x_stride + cc, mask=xm, other=0.0)
            tl.store(row.to(tl.pointer_type(tl.bfloat16)) + cc, xv.to(tl.bfloat16), mask=xm)
        k = tl.arange(0, K)
        e = tl.load(eid_ptr + src * K + k)
        p = tl.load(probs_ptr + src * p_stride + e).to(tl.float32)
        tl.store((row + po).to(tl.pointer_type(tl.float32)) + k, p)
        tl.store((row + io).to(tl.pointer_type(tl.int16)) + k, e.to(tl.int16))

    @triton.jit(do_not_specialize=["n_src", "H", "sx_stride", "sp_stride", "ox_stride", "op_stride"])
    def _gsum_packed_kernel(sx_ptr, sp_ptr, inv_ptr, gp_ptr, ox_ptr, op_ptr, n_src, H, sx_stride, sp_stride,
                            ox_stride, op_stride, W: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr,
                            MODE: tl.constexpr):
        # MODE 0: ox[r] = sum_w sx[inv[r, w]] (fp32 acc), op[r, :K] = sum_w sp[inv[r, w], :K]
        # MODE 1 (dispatch bwd, expert side): ox[r] = sum_w sx[inv[r, w]], op[r, w] = gp[inv[r, w]] (0 if none)
        r = tl.program_id(0)
        hb = tl.program_id(1)
        cols = hb * BLOCK + tl.arange(0, BLOCK)
        m = cols < H
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        kk = tl.arange(0, K)
        pacc = tl.zeros([K], dtype=tl.float32)
        for w in tl.static_range(W):
            j = tl.load(inv_ptr + r * W + w)
            if j < n_src:
                acc += tl.load(sx_ptr + j.to(tl.int64) * sx_stride + cols, mask=m, other=0.0).to(tl.float32)
                if hb == 0:
                    if MODE == 0:
                        pacc += tl.load(sp_ptr + j.to(tl.int64) * sp_stride + kk)
                    else:
                        g = tl.load(gp_ptr + j).to(tl.float32)
                        pacc += tl.where(kk == w, g, 0.0)
        tl.store(ox_ptr + r.to(tl.int64) * ox_stride + cols, acc.to(ox_ptr.dtype.element_ty), mask=m)
        if hb == 0:
            tl.store(op_ptr + r.to(tl.int64) * op_stride + kk, pacc)

    @triton.jit(do_not_specialize=["NN", "T"])
    def _plan_kernel(rm_ptr, idx_ptr, nn_ptr, ng_ptr, part_ptr, NN, T, NE: tl.constexpr, K: tl.constexpr,
                     E: tl.constexpr, NLE: tl.constexpr, G: tl.constexpr, NNP: tl.constexpr, BT: tl.constexpr):
        # BT tokens per program: expert ids (ascending), node / GPU need flags, and per-block partial counts
        # part[b] = [tokens per node (NN) | per GPU (E) | per expert (NE)]
        b = tl.program_id(0)
        t = b * BT + tl.arange(0, BT)
        tm = t < T
        e = tl.arange(0, NE)
        r = tl.load(rm_ptr + t.to(tl.int64)[:, None] * NE + e[None, :], mask=tm[:, None], other=0).to(tl.int32)
        r = tl.where(r != 0, 1, 0)
        c = tl.cumsum(r, 1) - 1
        tl.store(idx_ptr + t.to(tl.int64)[:, None] * K + c, (e[None, :] + 0 * c).to(tl.int64),
                 mask=(r > 0) & (c < K) & tm[:, None])
        gf = tl.max(tl.reshape(r, [BT, E, NLE]), 2)
        tl.store(ng_ptr + t.to(tl.int64)[:, None] * E + tl.arange(0, E)[None, :], gf.to(tl.int8), mask=tm[:, None])
        nf = tl.max(tl.reshape(gf, [BT, NNP, G]), 2)
        tl.store(nn_ptr + t.to(tl.int64)[:, None] * NN + tl.arange(0, NNP)[None, :], nf.to(tl.int8), mask=tm[:, None])
        W = NNP + E + NE
        tl.store(part_ptr + b * W + tl.arange(0, NNP), tl.sum(nf, 0))
        tl.store(part_ptr + b * W + NNP + tl.arange(0, E), tl.sum(gf, 0))
        tl.store(part_ptr + b * W + NNP + E + e, tl.sum(r, 0))

    # stable counting sort / compaction in two launches (replaces cumsum + where + scatter, or a radix sort)
    @triton.jit(do_not_specialize=["N", "R", "W", "KS", "KW", "KOFF", "VALID"])
    def _cs_hist_kernel(key_ptr, hist_ptr, N, R, W, KS, KW, KOFF, VALID, NBK: tl.constexpr, BLOCK: tl.constexpr, BOOL: tl.constexpr):
        b = tl.program_id(0)
        i = b * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        mem = (i % R) * W + i // R
        k = tl.load(key_ptr + (mem // KW) * KS + mem % KW, mask=m, other=0).to(tl.int32)
        if BOOL:
            k = 1 - k
        else:
            k = k - KOFF
            k = tl.where((k >= 0) & (k < VALID), k, VALID)
        k = tl.where(m, k, NBK)
        oh = (k[:, None] == tl.arange(0, NBK)[None, :]).to(tl.int32)
        tl.store(hist_ptr + b * NBK + tl.arange(0, NBK), tl.sum(oh, 0))

    @triton.jit(do_not_specialize=["N", "R", "W", "KS", "KW", "KOFF", "nblk", "VALID", "SENT", "MODR"])
    def _cs_scatter_kernel(key_ptr, hist_ptr, pos_ptr, perm_ptr, N, R, W, KS, KW, KOFF, nblk, VALID, SENT, MODR,
                           NBK: tl.constexpr, BLOCK: tl.constexpr, NBLKP: tl.constexpr, BOOL: tl.constexpr):
        b = tl.program_id(0)
        rows = tl.arange(0, NBLKP)
        cb = tl.arange(0, NBK)
        Hh = tl.load(hist_ptr + rows[:, None] * NBK + cb[None, :], mask=(rows < nblk)[:, None], other=0)
        tot = tl.sum(Hh, 0)
        prev = tl.sum(tl.where((rows < b)[:, None], Hh, 0), 0)
        base = tl.cumsum(tot, 0) - tot + prev
        i = b * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        mem = (i % R) * W + i // R
        k = tl.load(key_ptr + (mem // KW) * KS + mem % KW, mask=m, other=0).to(tl.int32)
        if BOOL:
            k = 1 - k
        else:
            k = k - KOFF
            k = tl.where((k >= 0) & (k < VALID), k, VALID)
        k = tl.where(m, k, NBK)
        oh = (k[:, None] == cb[None, :]).to(tl.int32)
        within = tl.cumsum(oh, 0) - oh
        pos = tl.sum(within * oh, 1) + tl.sum(oh * base[None, :], 1)
        valid = m & (k < VALID)
        tl.store(pos_ptr + mem, tl.where(valid, pos.to(tl.int64), SENT), mask=m)
        tl.store(perm_ptr + pos, (i % MODR).to(tl.int64), mask=valid)

    _HAVE_TRITON = os.environ.get("BEVERIN_T6_TRITON", "1") == "1"
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


def _gather_sum(src, inv, dtype):
    """out[r] = sum_w src[inv[r, w]] (inv == len(src) means none), accumulated in fp32, deterministic."""
    R, W = inv.shape
    Hd = src.shape[1]
    if _HAVE_TRITON and src.is_cuda and src.stride(1) == 1:
        out = torch.empty(R, Hd, dtype=dtype, device=src.device)
        if R and Hd:
            BLOCK = 1024 if Hd >= 1024 else triton.next_power_of_2(max(Hd, 16))
            nw = 4
            if _K3_GSUM and Hd <= 2048:  # bench2 K3 (k3-0004): whole row per program (1280 -> 2048), -22% at mid-4b
                BLOCK = triton.next_power_of_2(max(Hd, 16))
            _gather_sum_kernel[(R, triton.cdiv(Hd, BLOCK))](src, inv.contiguous(), out, src.shape[0], Hd,
                                                             src.stride(0), out.stride(0), W=W, BLOCK=BLOCK,
                                                             num_warps=nw)
        return out
    pad = torch.cat([src, src.new_zeros((1, Hd))], 0)
    acc = pad.index_select(0, inv[:, 0]).float()
    for w in range(1, W):
        acc += pad.index_select(0, inv[:, w])
    return acc.to(dtype)


def _quant_into(buf, x, idx, H):
    """Write fp8(x[idx]) into buf[:, :H] and the fp32 block scales into buf[:, H:H + 4H/128]."""
    if not len(idx):
        return
    if _HAVE_TRITON:
        _quant_gather_kernel[(len(idx),)](x, idx, buf, buf, x.stride(0), buf.stride(0), H, H, H // _QB,
                                          NBP=triton.next_power_of_2(H // _QB), QB=_QB,
                                          FMAX=240.0 if _fp8_dtype() == torch.float8_e4m3fnuz else 448.0)
        return
    dt = _fp8_dtype()
    xb = x.index_select(0, idx).float().view(-1, H // _QB, _QB)
    s = xb.abs().amax(-1).clamp(min=1e-20) / torch.finfo(dt).max
    buf[:, :H].copy_((xb / s[..., None]).to(dt).view(-1, H).view(torch.uint8))
    buf[:, H:H + 4 * (H // _QB)].view(torch.float32).copy_(s)


def _dequant_gather(buf, rows, H, dtype):
    out = torch.empty(len(rows), H, dtype=dtype, device=buf.device)
    if not len(rows):
        return out
    if _HAVE_TRITON:
        _dequant_gather_kernel[(len(rows),)](buf, rows, out, buf.stride(0), H, H, H // _QB,
                                            NBP=triton.next_power_of_2(H // _QB), QB=_QB)
        return out
    sel = buf.index_select(0, rows)
    q = sel[:, :H].contiguous().view(_fp8_dtype()).float().view(-1, H // _QB, _QB)
    s = sel[:, H:H + 4 * (H // _QB)].contiguous().view(torch.float32)
    return (q * s[..., None]).view(-1, H).to(dtype)


def _a2a(x, out_splits, in_splits, group):
    out = x.new_empty((int(sum(out_splits)),) + tuple(x.shape[1:]))
    dist.all_to_all_single(out, x.contiguous(), output_split_sizes=out_splits, input_split_sizes=in_splits,
                           group=group)
    return out


def _csort(key, R, W, nbk, valid, sent, modr, nperm, boolkey, ks=1, kw=1, koff=0, pos_shape=None):
    """Stable counting sort of N = R * W keys taken in (w, r) order from key[r * W + w]; returns pos (same layout as
    key, sent where key >= valid) and perm [nperm] (logical index % modr of each sorted valid slot)."""
    N = R * W
    BLOCK = 512 if nbk <= 64 else 256
    nblk = max(triton.cdiv(N, BLOCK), 1)
    dev = key.device
    hist = torch.empty(nblk, nbk, dtype=torch.int32, device=dev)
    pos = torch.empty(pos_shape if pos_shape is not None else key.shape, dtype=torch.long, device=dev)
    perm = torch.empty(max(nperm, 1), dtype=torch.long, device=dev)
    if N:
        _cs_hist_kernel[(nblk,)](key, hist, N, R, W, ks, kw, koff, valid, NBK=nbk, BLOCK=BLOCK, BOOL=boolkey)
        _cs_scatter_kernel[(nblk,)](key, hist, pos, perm, N, R, W, ks, kw, koff, nblk, valid, sent, modr, NBK=nbk, BLOCK=BLOCK,
                                    NBLKP=triton.next_power_of_2(nblk), BOOL=boolkey)
    return pos, perm[:nperm]


def _bucket(need, total):
    """need [R, W] bool -> (idx [total] source row of each slot in (w, r) order, inv [R, W] slot of (r, w) or total)."""
    R, W = need.shape
    if _FUSED and _HAVE_TRITON and need.is_cuda:
        inv, idx = _csort(need.contiguous().view(torch.int8), R, W, 2, 1, total, R, total, True)
        return idx, inv
    flat = need.t().reshape(-1)
    pos = torch.cumsum(flat, 0) - 1
    slot = torch.where(flat, pos, torch.full_like(pos, total))
    idx = torch.empty(total + 1, dtype=torch.long, device=need.device)
    src = torch.arange(R, device=need.device).repeat(W)
    idx.scatter_(0, slot, src)
    return idx[:total], slot.view(W, R).t().contiguous()


def _pad16(n):
    return (n + 15) // 16 * 16


# ------------------------------------------------------------------------------------------------------------------
_GROUPS = {}


def _pg_options(which):
    """Optional per-communicator RCCL CTA (channel) limits, e.g. BEVERIN_T6_RAIL_MAX_CTAS=8 to leave CUs to compute
    under EP overlap while the intra-node stage keeps all channels (a global NCCL_MAX_NCHANNELS=8 makes the node stage
    2.7x slower)."""
    mx = os.environ.get(f"BEVERIN_T6_{which}_MAX_CTAS")
    mn = os.environ.get(f"BEVERIN_T6_{which}_MIN_CTAS")
    if not (mx or mn):
        return None
    opts = dist.ProcessGroupNCCL.Options()
    if mx:
        opts.config.max_ctas = int(mx)
    if mn:
        opts.config.min_ctas = int(mn)
    return opts


def _subgroups(ep_group, gpn):
    """(rail group, node group) for this rank, cached per EP group. Only the member ranks call new_group
    (use_local_synchronization), so pipeline stages with different layer counts do not have to agree."""
    ranks = dist.get_process_group_ranks(ep_group)
    key = tuple(ranks)
    if key not in _GROUPS:
        me = dist.get_rank(ep_group)
        node, local = divmod(me, gpn)
        rail = [ranks[m * gpn + local] for m in range(len(ranks) // gpn)]
        nodeg = ranks[node * gpn:(node + 1) * gpn]
        g_rail = dist.new_group(rail, use_local_synchronization=True, pg_options=_pg_options("RAIL"))
        g_node = dist.new_group(nodeg, use_local_synchronization=True, pg_options=_pg_options("NODE"))
        # connect now (CXI endpoints must be created before memory use grows above ~85 GiB per APU)
        t = torch.zeros(len(rail), dtype=torch.float32, device="cuda")
        dist.all_to_all_single(torch.empty_like(t), t, group=g_rail)
        t = torch.zeros(gpn, dtype=torch.float32, device="cuda")
        dist.all_to_all_single(torch.empty_like(t), t, group=g_node)
        torch.cuda.synchronize()
        _GROUPS[key] = (g_rail, g_node)
    return _GROUPS[key]


class HierEP:
    """Hierarchical dispatch/combine for one EP group laid out as `gpn` consecutive EP ranks per node."""

    def __init__(self, ep_group, num_experts, topk, hidden, gpn=4):
        self.group = ep_group
        self.E = dist.get_world_size(ep_group)
        self.me = dist.get_rank(ep_group)
        self.G = gpn
        assert self.E % gpn == 0, (self.E, gpn)
        self.NN = self.E // gpn
        self.node, self.local = divmod(self.me, gpn)
        self.NE, self.K, self.H = num_experts, topk, hidden
        self.NLE = num_experts // self.E
        import socket, zlib
        h = torch.tensor([zlib.crc32(socket.gethostname().encode())], dtype=torch.long, device="cuda")
        hs = torch.empty(self.E, dtype=torch.long, device="cuda")
        dist.all_gather_into_tensor(hs, h, group=ep_group)
        hs = hs.tolist()
        for r in range(self.E):
            assert hs[r] == hs[(r // gpn) * gpn], f"EP rank {r} is not on the node of EP rank {(r // gpn) * gpn}"
        assert len(set(hs)) == self.NN, "EP group must span exactly E/gpn nodes"
        self.g_rail, self.g_node = _subgroups(ep_group, gpn)
        H, K = hidden, topk
        self.fp8 = _FP8 and H % _QB == 0
        # row layout of the packed dispatch buffer (bytes)
        self.xb = H + 4 * (H // _QB) if self.fp8 else 2 * H
        self.po = _pad16(self.xb)
        self.io = self.po + 4 * K
        self.RB = _pad16(self.io + 2 * K)

    # ---------------------------------------------------------------------------------------------------------
    def plan(self, routing_map):
        with _tm("p_plan"):
            E, G, NN, NLE, K = self.E, self.G, self.NN, self.NLE, self.K
            T = routing_map.shape[0]
            pk = (_FUSED and _HAVE_TRITON and routing_map.is_cuda and NN & (NN - 1) == 0 and E & (E - 1) == 0
                  and NLE & (NLE - 1) == 0)
            if pk:
                rmi = routing_map.contiguous().view(torch.int8) if routing_map.dtype == torch.bool else \
                    (routing_map != 0).contiguous().view(torch.int8)
                idx = torch.empty(T, K, dtype=torch.long, device=rmi.device)
                need_node = torch.empty(T, NN, dtype=torch.bool, device=rmi.device)
                need_gpu = torch.empty(T, E, dtype=torch.bool, device=rmi.device)
                BT = 32
                nb = max(triton.cdiv(T, BT), 1)
                part = torch.zeros(nb, NN + E + self.NE, dtype=torch.int32, device=rmi.device)
                if T:
                    _plan_kernel[(nb,)](rmi, idx, need_node.view(torch.int8), need_gpu.view(torch.int8), part, NN, T,
                                        NE=self.NE, K=K, E=E, NLE=NLE, G=G, NNP=NN, BT=BT)
                vec = part.sum(0, dtype=torch.long)
            elif _K3_TOPK and routing_map.shape[1] <= 1024:  # bench2 K3 (k3-0002): same id set, ascending, one kernel
                from megatron.core.transformer.moe.beverin_k3 import mask_to_idx
                idx = mask_to_idx(routing_map, K)
            else:
                idx = torch.topk(routing_map.to(torch.int8), K, dim=1, sorted=False).indices  # [T, K] expert ids
            gpu = idx // NLE
            if not pk:
                node = gpu // G
                need_node = torch.zeros(T, NN, dtype=torch.bool, device=idx.device).scatter_(1, node, True)
                need_gpu = torch.zeros(T, E, dtype=torch.bool, device=idx.device).scatter_(1, gpu, True)
            if pk:
                pass
            elif os.environ.get("BEVERIN_T6_GEMV", "0") == "1":  # slower (681685): 0.47 -> 0.73 ms
                cat = torch.cat([need_node, need_gpu, routing_map.bool()], 1).to(torch.float32)
                vec = torch.ones(1, T, dtype=torch.float32, device=idx.device).matmul(cat).view(-1).round().long()
            else:
                vec = torch.cat([need_node.sum(0), need_gpu.sum(0), routing_map.sum(0)]).long()
            allv = torch.empty(E, vec.numel(), dtype=torch.long, device=vec.device)
            dist.all_gather_into_tensor(allv, vec, group=self.group)
            C = allv.cpu()  # sync
            Cn, Cg, Ce = C[:, :NN], C[:, NN:NN + E], C[:, NN + E:]
            rail = [m * G + self.local for m in range(NN)]
            mine = [self.node * G + j for j in range(G)]
            in1 = [int(Cn[self.me, m]) for m in range(NN)]
            out1 = [int(Cn[rail[m], self.node]) for m in range(NN)]
            in2 = [int(sum(Cg[s, mine[j]] for s in rail)) for j in range(G)]
            out2 = [int(sum(Cg[m * G + j, self.me] for m in range(NN))) for j in range(G)]
            tpe = Ce[:, self.me * NLE:(self.me + 1) * NLE].sum(0)
            tpe_dev = allv[:, NN + E + self.me * NLE:NN + E + (self.me + 1) * NLE].sum(0) if _DEVTPE else None
            return dict(idx=idx, need_node=need_node, in1=in1, out1=out1, in2=in2, out2=out2, tpe=tpe, tpe_dev=tpe_dev,
                        T=T)

    def dispatch(self, x, probs, pl):
        with _tm("d_total"):
            x3, p3 = _Dispatch.apply(x, probs, self, pl)
        tpe = pl["tpe"].clone()
        if pl.get("tpe_dev") is not None:
            tpe._beverin_dev = pl["tpe_dev"]
        return x3, p3, tpe

    def combine(self, y3, pl):
        with _tm("c_total"):
            return _Combine.apply(y3, self, pl)

    # ---------------------------------------------------------------------------------------------------------
    def _dispatch_fwd(self, x, probs, pl):
        G, NLE, K, H, RB = self.G, self.NLE, self.K, self.H, self.RB
        dev = x.device
        idx = pl["idx"]
        with _tm("d1_pack"):
            S1 = sum(pl["in1"])
            idx1, inv1 = _bucket(pl["need_node"], S1)
            buf1 = torch.empty(S1, RB, dtype=torch.uint8, device=dev)
            if _FUSED and _HAVE_TRITON and S1:
                pr = probs if probs.stride(1) == 1 else probs.contiguous()
                _pack1_kernel[(S1,)](x, idx1, idx.contiguous(), pr, buf1, x.stride(0), pr.stride(0), RB, H, self.po,
                                     self.io, H, H // _QB, K=K, NBP=triton.next_power_of_2(max(H // _QB, 1)), QB=_QB,
                                     FMAX=240.0 if _fp8_dtype() == torch.float8_e4m3fnuz else 448.0, FP8=self.fp8,
                                     HB=triton.next_power_of_2(H))
            elif self.fp8:
                _quant_into(buf1, x, idx1, H)
            else:
                buf1[:, :2 * H].view(x.dtype).copy_(x.index_select(0, idx1))
            if not (_FUSED and _HAVE_TRITON and S1):
                buf1[:, self.po:self.po + 4 * K].view(torch.float32).copy_(probs.gather(1, idx).float().index_select(0, idx1))
                buf1[:, self.io:self.io + 2 * K].view(torch.int16).copy_(idx.index_select(0, idx1).to(torch.int16))
        with _tm("d1_a2a_rail"):
            r1 = _a2a(buf1, pl["out1"], pl["in1"], self.g_rail)
        del buf1
        with _tm("d2_pack"):
            R1 = r1.shape[0]
            i1 = r1[:, self.io:self.io + 2 * K].view(torch.int16).long()
            g_loc = i1 // NLE - self.node * G
            g_loc = torch.where((g_loc >= 0) & (g_loc < G), g_loc, torch.full_like(g_loc, G))
            need2 = torch.zeros(R1, G + 1, dtype=torch.bool, device=dev).scatter_(1, g_loc, True)[:, :G]
            S2 = sum(pl["in2"])
            idx2, inv2 = _bucket(need2, S2)
            buf2 = r1.index_select(0, idx2)
        del r1
        with _tm("d2_a2a_node"):
            r2 = _a2a(buf2, pl["out2"], pl["in2"], self.g_node)
        del buf2
        with _tm("d3_unpack"):
            R2 = r2.shape[0]
            if not (_FUSED and _HAVE_TRITON):
                e_loc = r2[:, self.io:self.io + 2 * K].view(torch.int16).long() - self.me * NLE
                e_loc = torch.where((e_loc >= 0) & (e_loc < NLE), e_loc, torch.full_like(e_loc, NLE))
            N3 = int(pl["tpe"].sum())
            if _FUSED and _HAVE_TRITON:
                ids = r2.view(torch.int16)  # [R2, RB / 2]; expert ids at columns io / 2 .. io / 2 + K
                inv3f, order = _csort(ids[:, self.io // 2:], R2 * K, 1, triton.next_power_of_2(NLE + 1), NLE, N3,
                                      1 << 62, N3, False, ks=self.RB // 2, kw=K, koff=self.me * NLE,
                                      pos_shape=(R2 * K,))
            elif _FUSED:
                order = torch.sort(e_loc.reshape(-1).to(torch.int32), stable=True).indices[:N3]
            else:
                q = torch.arange(R2 * K, device=dev)
                order = torch.sort(e_loc.reshape(-1) * (R2 * K) + q).indices[:N3]
            if _FUSED and _HAVE_TRITON:
                inv3 = inv3f
            else:
                inv3 = torch.full((R2 * K,), N3, dtype=torch.long, device=dev)
                inv3[order] = torch.arange(N3, device=dev)
            rows3 = order // K
            if self.fp8:
                x3 = _dequant_gather(r2, rows3, H, x.dtype)
            else:
                x3 = r2[:, :2 * H].view(x.dtype).index_select(0, rows3)
            p3 = r2[:, self.po:self.po + 4 * K].view(torch.float32).reshape(-1).index_select(0, order)
        pl.update(idx1=idx1, inv1=inv1, idx2=idx2, inv2=inv2, rows3=rows3, inv3=inv3.view(R2, K), order=order,
                  R1=R1, R2=R2, xdt=x.dtype, pdt=probs.dtype)
        if _STASH:
            pl["r2"] = r2
        return x3, p3

    def rebuild_x3(self, r2, rows3, dtype):
        """The expert-sorted dispatch output x3 from the node-stage receive buffer (same kernel as the forward)."""
        if self.fp8:
            return _dequant_gather(r2, rows3, self.H, dtype)
        return r2[:, :2 * self.H].view(dtype).index_select(0, rows3)

    def _dispatch_bwd(self, gx3, gp3, pl):
        """Mirror of the dispatch: sum the copies back to the source tokens. Returns (grad x [T, H], grad probs)."""
        K, H = self.K, self.H
        dev = gx3.device
        xdt = pl["xdt"]
        R2 = pl["R2"]
        fz = _FUSED and _HAVE_TRITON and gx3.is_cuda and gx3.stride(1) == 1
        BLK = 1024 if H >= 1024 else 256
        if fz:
            with _tm("db3"):
                b = torch.empty(R2, 2 * H + 4 * K, dtype=torch.uint8, device=dev)
                bx, bp = b[:, :2 * H].view(xdt), b[:, 2 * H:].view(torch.float32)
                if R2:
                    _gsum_packed_kernel[(R2, triton.cdiv(H, BLK))](gx3, gx3, pl["inv3"], gp3.float().contiguous(), bx, bp,
                                                                  gx3.shape[0], H, gx3.stride(0), 0, bx.stride(0),
                                                                  bp.stride(0), W=K, K=K, BLOCK=BLK, MODE=1)
            with _tm("db2_a2a_node"):
                b = _a2a(b, pl["in2"], pl["out2"], self.g_node)
            with _tm("db2_sum"):
                R1 = pl["inv2"].shape[0]
                b2 = torch.empty(R1, 2 * H + 4 * K, dtype=torch.uint8, device=dev)
                sx, sp = b[:, :2 * H].view(xdt), b[:, 2 * H:].view(torch.float32)
                ox, op = b2[:, :2 * H].view(xdt), b2[:, 2 * H:].view(torch.float32)
                W2 = pl["inv2"].shape[1]
                if R1:
                    _gsum_packed_kernel[(R1, triton.cdiv(H, BLK))](sx, sp, pl["inv2"], sp, ox, op, b.shape[0], H,
                                                                  sx.stride(0), sp.stride(0), ox.stride(0), op.stride(0),
                                                                  W=W2, K=K, BLOCK=BLK, MODE=0)
            with _tm("db1_a2a_rail"):
                b = _a2a(b2, pl["in1"], pl["out1"], self.g_rail)
            with _tm("db1_sum"):
                T = pl["T"]
                gx = torch.empty(T, H, dtype=xdt, device=dev)
                gpk = torch.empty(T, K, dtype=torch.float32, device=dev)
                sx, sp = b[:, :2 * H].view(xdt), b[:, 2 * H:].view(torch.float32)
                W1 = pl["inv1"].shape[1]
                if T:
                    _gsum_packed_kernel[(T, triton.cdiv(H, BLK))](sx, sp, pl["inv1"], sp, gx, gpk, b.shape[0], H,
                                                                 sx.stride(0), sp.stride(0), gx.stride(0), gpk.stride(0),
                                                                 W=W1, K=K, BLOCK=BLK, MODE=0)
                gprobs = torch.zeros(T, self.NE, dtype=pl["pdt"], device=dev).scatter_(1, pl["idx"], gpk.to(pl["pdt"]))
            return gx, gprobs
        with _tm("db3"):
            g2 = _gather_sum(gx3, pl["inv3"], xdt)  # [R2, H]
            gp2 = torch.zeros(R2 * K, dtype=torch.float32, device=dev)
            gp2[pl["order"]] = gp3.float()
            b = torch.empty(R2, 2 * H + 4 * K, dtype=torch.uint8, device=dev)
            b[:, :2 * H].view(xdt).copy_(g2)
            b[:, 2 * H:].view(torch.float32).copy_(gp2.view(R2, K))
        with _tm("db2_a2a_node"):
            b = _a2a(b, pl["in2"], pl["out2"], self.g_node)
        with _tm("db2_sum"):
            g1 = _gather_sum(b[:, :2 * H].view(xdt), pl["inv2"], xdt)
            gp1 = _gather_sum(b[:, 2 * H:].view(torch.float32), pl["inv2"], torch.float32)
            b = torch.empty(g1.shape[0], 2 * H + 4 * K, dtype=torch.uint8, device=dev)
            b[:, :2 * H].view(xdt).copy_(g1)
            b[:, 2 * H:].view(torch.float32).copy_(gp1)
        with _tm("db1_a2a_rail"):
            b = _a2a(b, pl["in1"], pl["out1"], self.g_rail)
        with _tm("db1_sum"):
            gx = _gather_sum(b[:, :2 * H].view(xdt), pl["inv1"], xdt)
            gpk = _gather_sum(b[:, 2 * H:].view(torch.float32), pl["inv1"], torch.float32)
            gprobs = torch.zeros(pl["T"], self.NE, dtype=pl["pdt"], device=dev).scatter_(1, pl["idx"],
                                                                                         gpk.to(pl["pdt"]))
        return gx, gprobs

    def _combine_fwd(self, y3, pl):
        dt = y3.dtype
        with _tm("c3_sum"):
            y2 = _gather_sum(y3, pl["inv3"], dt)
        with _tm("c2_a2a_node"):
            y2 = _a2a(y2, pl["in2"], pl["out2"], self.g_node)
        with _tm("c2_sum"):
            y1 = _gather_sum(y2, pl["inv2"], dt)
        del y2
        with _tm("c1_a2a_rail"):
            y1 = _a2a(y1, pl["in1"], pl["out1"], self.g_rail)
        with _tm("c1_sum"):
            return _gather_sum(y1, pl["inv1"], dt)

    def _combine_bwd(self, g, pl):
        with _tm("cb1_a2a_rail"):
            g = _a2a(g.index_select(0, pl["idx1"]), pl["out1"], pl["in1"], self.g_rail)
        with _tm("cb2_a2a_node"):
            g = _a2a(g.index_select(0, pl["idx2"]), pl["out2"], pl["in2"], self.g_node)
        with _tm("cb3"):
            return g.index_select(0, pl["rows3"])


class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probs, hier, pl):
        ctx.hier, ctx.pl = hier, pl
        return hier._dispatch_fwd(x, probs, pl)

    @staticmethod
    def backward(ctx, gx3, gp3):
        dev = (gx3 if gx3 is not None else gp3).device
        pl = ctx.pl
        if gx3 is None:
            gx3 = torch.zeros(len(pl["rows3"]), ctx.hier.H, dtype=pl["xdt"], device=dev)
        if gp3 is None:
            gp3 = torch.zeros(len(pl["rows3"]), dtype=torch.float32, device=dev)
        gx, gp = ctx.hier._dispatch_bwd(gx3.contiguous(), gp3.contiguous(), pl)
        return gx, gp, None, None


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y3, hier, pl):
        ctx.hier, ctx.pl = hier, pl
        return hier._combine_fwd(y3.contiguous(), pl)

    @staticmethod
    def backward(ctx, g):
        return ctx.hier._combine_bwd(g.contiguous(), ctx.pl), None, None
