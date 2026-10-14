# beverin lab2 A3: hierarchical, rail-aligned EP dispatch/combine without host counts on the critical path.
#
# Successor of round 1's T6 path (lab/t6/src/beverin_ep_t6.py), same routing semantics, different mechanics:
#   - stage 1 (inter-node, RCCL on a 4-rank rail communicator): every token goes to every rail peer (fixed size T rows,
#     "padded" rail; about 10% more bytes than sending only the tokens a node needs, but no split sizes are needed, so
#     nothing on the host waits before the exchange is queued). Pairwise rounds (round r: send to node+r, receive from
#     node-r), so the intra-node stage of round r runs while round r+1 is on the wire.
#   - stage 2 (intra-node, xGMI): no RCCL. The holder of a row (the GPU that received it over its rail) writes each copy
#     straight into the expert-sorted input buffer of the node-mate that owns the expert ("push", Triton kernel with a
#     chosen number of CTAs), at a position computed on the device from the all-gathered per-rank expert counts:
#     x3 order = (local expert, source EP rank, token), the same order as Megatron's alltoall dispatcher. Buffers are
#     torch symmetric memory on the node group (a ring of 2 slots, one device barrier per exchange).
#   - the reverse direction (combine, dispatch backward) pulls the expert outputs from the node-mates' slots, sums the
#     copies of a row in fp32 (fixed order, deterministic) and sends the partial sums back over the rail.
#   - fp8 forward dispatch (e4m3, one fp32 scale per 128 columns) as round 1; combine and backward bf16.
# Host syncs: one event wait per dispatch, after the whole exchange is queued (for N3 and tokens per expert, which the
# expert GEMM needs on the host). Combine and both backward exchanges have none.
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

_QB = 128
_TIMING = os.environ.get("A3_TIMING", "0") == "1"
_DEBUG = os.environ.get("A3_DEBUG", "0") == "1"


def _dbg(*a):
    if _DEBUG:
        torch.cuda.synchronize()
        print(f"{{\"dbg\": \"r{dist.get_rank()}\"}}", *a, flush=True)
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
    out = {k: sorted(s.elapsed_time(e) for s, e in v)[len(v) // 2] for k, v in TIMES.items()}
    if reset:
        TIMES.clear()
    return out


def _pad16(n):
    return (n + 15) // 16 * 16


def _fp8_dtype():
    arch = getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "gcnArchName", "")
    return torch.float8_e4m3fnuz if "gfx94" in arch else torch.float8_e4m3fn


_POSBITS = 27  # posk code: (j << 27) | pos, -1 = copy not on this node


# ----------------------------------------------------------------------------------------------------------------
# kernels
@triton.jit
def _pack_kernel(x_ptr, probs_ptr, idx_ptr, rk_ptr, out_ptr, H, NE, x_stride, RB, s_off, p_off, c_off,
                 K: tl.constexpr, FP8: tl.constexpr, NBP: tl.constexpr, QB: tl.constexpr, FMAX: tl.constexpr,
                 HP: tl.constexpr):
    # one program per token t: row = [x (fp8 + fp32 scales per QB, or bf16) | probs fp32 K | codes int32 K]
    t = tl.program_id(0).to(tl.int64)
    row = out_ptr + t * RB
    if FP8:
        NB = H // QB
        b = tl.arange(0, NBP)
        c = tl.arange(0, QB)
        bm = b < NB
        cols = b[:, None] * QB + c[None, :]
        m = bm[:, None] & (c[None, :] < QB)
        x = tl.load(x_ptr + t * x_stride + cols, mask=m, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=1)
        s = tl.maximum(amax, 1e-20) / FMAX
        q = (x / s[:, None]).to(tl.float8e4b8)
        tl.store(row + cols, q.to(tl.uint8, bitcast=True), mask=m)
        tl.store((row + s_off).to(tl.pointer_type(tl.float32)) + b, s, mask=bm)
    else:
        cols = tl.arange(0, HP)
        m = cols < H
        x = tl.load(x_ptr + t * x_stride + cols, mask=m)
        tl.store(row.to(tl.pointer_type(tl.bfloat16)) + cols, x, mask=m)
    k = tl.arange(0, K)
    e = tl.load(idx_ptr + t * K + k)
    p = tl.load(probs_ptr + t * NE + e).to(tl.float32)
    r = tl.load(rk_ptr + t * K + k)
    tl.store((row + p_off).to(tl.pointer_type(tl.float32)) + k, p)
    tl.store((row + c_off).to(tl.pointer_type(tl.int32)) + k, (e.to(tl.int32) << 20) | r)


@triton.jit(do_not_specialize=["R", "s_idx", "slot_ptr"])
def _push_fwd_kernel(rows_ptr, R, RB, p_off, c_off, NW, b_ptr, s_idx, ebase, NLE,
                     slot_ptr, dtab_ptr, ORB, posk_ptr, K: tl.constexpr, NWP: tl.constexpr, NEN: tl.constexpr,
                     POSBITS: tl.constexpr):
    # rows from source EP rank s_idx: each copy (row r, k) whose expert e is on this node goes to node-mate
    # j = (e - ebase) // NLE at position B[s, e - ebase] + rank. Destination row = [payload NW words | prob fp32].
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    w = tl.arange(0, NWP)
    wm = w < NW
    for r in range(pid, R, npg):
        row = rows_ptr + r.to(tl.int64) * RB
        v = tl.load(row.to(tl.pointer_type(tl.int32)) + w, mask=wm)
        for k in tl.static_range(K):
            code = tl.load((row + c_off).to(tl.pointer_type(tl.int32)) + k)
            el = (code >> 20) - ebase
            ok = (el >= 0) & (el < NEN)
            pk = -1
            if ok:
                j = el // NLE
                pos = tl.load(b_ptr + s_idx * NEN + el) + (code & 0xFFFFF)
                pk = (j << POSBITS) | pos
                dst = (slot_ptr + tl.load(dtab_ptr + j) + pos.to(tl.int64) * ORB).to(tl.pointer_type(tl.uint8))
                tl.store(dst.to(tl.pointer_type(tl.int32)) + w, v, mask=wm)
                pr = tl.load((row + p_off).to(tl.pointer_type(tl.float32)) + k)
                tl.store((dst + NW * 4).to(tl.pointer_type(tl.float32)), pr)
            tl.store(posk_ptr + r.to(tl.int64) * K + k, pk)


@triton.jit(do_not_specialize=["R", "slot_ptr"])
def _push_bwd_kernel(rows_ptr, R, RB, NW, posk_ptr, slot_ptr, dtab_ptr, ORB, K: tl.constexpr, NWP: tl.constexpr,
                     POSBITS: tl.constexpr):
    # combine backward: row r (bf16 grad, NW words) goes to every copy position of posk[r]
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    w = tl.arange(0, NWP)
    wm = w < NW
    for r in range(pid, R, npg):
        v = tl.load((rows_ptr + r.to(tl.int64) * RB).to(tl.pointer_type(tl.int32)) + w, mask=wm)
        for k in tl.static_range(K):
            pk = tl.load(posk_ptr + r.to(tl.int64) * K + k)
            if pk >= 0:
                j = pk >> POSBITS
                pos = pk & ((1 << POSBITS) - 1)
                dst = (slot_ptr + tl.load(dtab_ptr + j) + pos.to(tl.int64) * ORB).to(tl.pointer_type(tl.uint8))
                tl.store(dst.to(tl.pointer_type(tl.int32)) + w, v, mask=wm)


@triton.jit(do_not_specialize=["R", "slot_ptr"])
def _pull_sum_kernel(posk_ptr, R, slot_ptr, dtab_ptr, YRB, gp_off, out_ptr, ORB, H, K: tl.constexpr,
                     HP: tl.constexpr, WITH_GP: tl.constexpr, POSBITS: tl.constexpr):
    # out row r = [sum over copies k of Y_j[pos] (bf16, fp32 accumulation) | gp_k fp32 K (if WITH_GP)]
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    cols = tl.arange(0, HP)
    cm = cols < H
    kk = tl.arange(0, K)
    for r in range(pid, R, npg):
        acc = tl.zeros([HP], dtype=tl.float32)
        gpv = tl.zeros([K], dtype=tl.float32)
        for k in tl.static_range(K):
            pk = tl.load(posk_ptr + r.to(tl.int64) * K + k)
            if pk >= 0:
                j = pk >> POSBITS
                pos = pk & ((1 << POSBITS) - 1)
                src = (slot_ptr + tl.load(dtab_ptr + j) + pos.to(tl.int64) * YRB).to(tl.pointer_type(tl.uint8))
                acc += tl.load(src.to(tl.pointer_type(tl.bfloat16)) + cols, mask=cm, other=0.0).to(tl.float32)
                if WITH_GP:
                    g = tl.load((src + gp_off).to(tl.pointer_type(tl.float32)))
                    gpv = tl.where(kk == k, g, gpv)
        orow = out_ptr + r.to(tl.int64) * ORB
        tl.store(orow.to(tl.pointer_type(tl.bfloat16)) + cols, acc.to(tl.bfloat16), mask=cm)
        if WITH_GP:
            tl.store((orow + 2 * H).to(tl.pointer_type(tl.float32)) + kk, gpv)


@triton.jit
def _sumn_kernel(p0, PS, NP, RB, out_ptr, gp_ptr, H, K: tl.constexpr, HP: tl.constexpr, WITH_GP: tl.constexpr):
    # out[t] = sum over parts i (in order) of part_i[t]; part_i = p0 + i * PS (bf16 rows of RB bytes, fp32 accumulation)
    t = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, HP)
    cm = cols < H
    kk = tl.arange(0, K)
    a = tl.zeros([HP], dtype=tl.float32)
    g = tl.zeros([K], dtype=tl.float32)
    for i in range(NP):
        row = p0 + i * PS + t * RB
        a += tl.load(row.to(tl.pointer_type(tl.bfloat16)) + cols, mask=cm, other=0.0).to(tl.float32)
        if WITH_GP:
            g += tl.load((row + 2 * H).to(tl.pointer_type(tl.float32)) + kk)
    tl.store(out_ptr + t * H + cols, a.to(tl.bfloat16), mask=cm)
    if WITH_GP:
        tl.store(gp_ptr + t * K + kk, g)


@triton.jit(do_not_specialize=["N"])
def _dequant_kernel(slot_ptr, ORB, N, x_ptr, p_ptr, H, s_off, pr_off, NBP: tl.constexpr, QB: tl.constexpr,
                    FP8: tl.constexpr, HP: tl.constexpr):
    # x3[n] = dequant(slot row n), p3[n] = prob field
    n = tl.program_id(0).to(tl.int64)
    row = slot_ptr + n * ORB
    if FP8:
        NB = H // QB
        b = tl.arange(0, NBP)
        c = tl.arange(0, QB)
        bm = b < NB
        cols = b[:, None] * QB + c[None, :]
        m = bm[:, None] & (c[None, :] < QB)
        q = tl.load(row + cols, mask=m, other=0).to(tl.float8e4b8, bitcast=True).to(tl.float32)
        s = tl.load((row + s_off).to(tl.pointer_type(tl.float32)) + b, mask=bm, other=0.0)
        tl.store(x_ptr + n * H + cols, (q * s[:, None]).to(tl.bfloat16), mask=m)
    else:
        cols = tl.arange(0, HP)
        m = cols < H
        tl.store(x_ptr + n * H + cols, tl.load(row.to(tl.pointer_type(tl.bfloat16)) + cols, mask=m), mask=m)
    tl.store(p_ptr + n, tl.load((row + pr_off).to(tl.pointer_type(tl.float32))))


# ----------------------------------------------------------------------------------------------------------------
_GROUPS = {}


def _pg_options(which):
    mx = os.environ.get(f"A3_{which}_MAX_CTAS")
    if not mx:
        return None
    opts = dist.ProcessGroupNCCL.Options()
    opts.config.max_ctas = int(mx)
    return opts


def _subgroups(ep_group, gpn):
    ranks = dist.get_process_group_ranks(ep_group)
    key = tuple(ranks)
    if key not in _GROUPS:
        me = dist.get_rank(ep_group)
        node, local = divmod(me, gpn)
        rail = [ranks[m * gpn + local] for m in range(len(ranks) // gpn)]
        nodeg = ranks[node * gpn:(node + 1) * gpn]
        g_rail = dist.new_group(rail, use_local_synchronization=True, pg_options=_pg_options("RAIL"))
        g_node = dist.new_group(nodeg, use_local_synchronization=True)
        t = torch.zeros(len(rail), dtype=torch.float32, device="cuda")
        dist.all_to_all_single(torch.empty_like(t), t, group=g_rail)
        if len(rail) > 1:
            ops = [dist.P2POp(dist.isend, t[:1], rail[(node + 1) % len(rail)], g_rail),
                   dist.P2POp(dist.irecv, t[1:2], rail[(node - 1) % len(rail)], g_rail)]
            for w in dist.batch_isend_irecv(ops):
                w.wait()
        t = torch.zeros(gpn, dtype=torch.float32, device="cuda")
        dist.all_reduce(t, group=g_node)
        torch.cuda.synchronize()
        _GROUPS[key] = (g_rail, g_node, rail, nodeg)
    return _GROUPS[key]


class _Ring:
    """Two slots of node-shared (symmetric) memory used by every A3 exchange object of a node group (all MoE layers
    share them): each exchange takes the next slot and has exactly one device barrier between its writes and the
    reads by the node-mates, so the slot of exchange k is rewritten (exchange k + 2) only after every node-mate has
    passed the barrier of exchange k + 1, i.e. finished its reads of exchange k. Requires all exchanges of a rank
    on one stream, in the same order on all node-mates (as RCCL requires anyway)."""

    def __init__(self, g_node, gpn, slot_bytes):
        import torch.distributed._symmetric_memory as symm_mem
        self.slot_bytes = slot_bytes
        self.buf = symm_mem.empty(2 * slot_bytes, dtype=torch.uint8, device="cuda")
        self.hdl = symm_mem.rendezvous(self.buf, g_node.group_name)
        base = self.buf.data_ptr()
        self.dtab = torch.tensor([self.hdl.get_buffer(r, (2 * slot_bytes,), torch.uint8).data_ptr() - base
                                  for r in range(gpn)], dtype=torch.int64, device="cuda")
        self.n = 0
        # every slot access (pushes, barrier, local reads, y copies, pulls) runs on this one stream, so the slot
        # protocol holds even when the caller issues exchanges on different streams (Megatron's EP-overlap schedule
        # uses the compute stream in warm-up/flush and its comm stream in the steady state)
        self.stream = torch.cuda.Stream()

    def next(self):
        s = self.n
        self.n ^= 1
        return s * self.slot_bytes


_RINGS = {}


def _ring(g_node, gpn, slot_bytes):
    """The node group's ring, created or grown (collective over the node group: every node-mate makes the same call
    at the same exchange, the sizes come from all-gathered counts). A grown ring replaces the old one for every
    exchange object; the old one is kept alive (kernels in flight may still use it)."""
    key = tuple(dist.get_process_group_ranks(g_node))
    r = _RINGS.get(key)
    if r is None or r.slot_bytes < slot_bytes:
        new = _Ring(g_node, gpn, slot_bytes)
        if r is not None:
            new.old = r
            new.n = r.n
            new.stream = r.stream
        _RINGS[key] = r = new
    return r


class A3EP:
    """Hierarchical dispatch/combine for one EP group laid out as `gpn` consecutive EP ranks per node."""

    def __init__(self, ep_group, num_experts, topk, hidden, gpn=4, fp8=None, cap_rows=None):
        import torch.distributed._symmetric_memory as symm_mem
        self.group = ep_group
        self.E = dist.get_world_size(ep_group)
        self.me = dist.get_rank(ep_group)
        self.G = gpn
        self.NN = self.E // gpn
        self.node, self.local = divmod(self.me, gpn)
        self.NE, self.K, self.H = num_experts, topk, hidden
        self.NLE = num_experts // self.E
        self.NEN = self.NLE * gpn  # experts per node
        assert self.NEN <= 4096 and self.K <= 16
        _dbg("init: subgroups")
        self.g_rail, self.g_node, self.rail_ranks, self.node_ranks = _subgroups(ep_group, gpn)
        _dbg("init: subgroups done")
        H, K = hidden, topk
        self.fp8 = (os.environ.get("A3_FP8", "0") == "1" if fp8 is None else fp8) and H % _QB == 0
        # dispatch row (rail): [x | probs fp32 K | codes int32 K]
        self.xb = H + 4 * (H // _QB) if self.fp8 else 2 * H
        self.s_off = H  # fp8 scales
        self.p_off = _pad16(self.xb)
        self.c_off = self.p_off + 4 * K
        self.RB = _pad16(self.c_off + 4 * K)
        # dispatch row (node slot): [x payload (xb, padded to words) | prob fp32]
        self.NW = (self.xb + 3) // 4
        self.XRB = _pad16(self.NW * 4 + 4)
        # reverse direction slot row: [y bf16 H | gp fp32]
        self.YRB = _pad16(2 * H + 4)
        self.GRB = 2 * H  # combine-backward slot row (bf16 grad)
        self.CTAS = int(os.environ.get("A3_CTAS", 64))
        self.RAIL_MODE = os.environ.get("A3_RAIL", "rounds")  # rounds | allgather
        T_hint = int(os.environ.get("A3_T", 8192))
        cap = cap_rows or int(os.environ.get("A3_CAP", 0)) or int(float(os.environ.get("A3_CAP_FACTOR", 2.0)) * T_hint * K)
        self.ring = _ring(self.g_node, gpn, cap * max(self.XRB, self.YRB, self.GRB))
        self.cap = self.ring.slot_bytes // max(self.XRB, self.YRB, self.GRB)
        self.slot_bytes = self.ring.slot_bytes
        self.buf, self.hdl, self.dtab = self.ring.buf, self.ring.hdl, self.ring.dtab
        self.sr = torch.cuda.Stream()
        self.fmax = 240.0 if _fp8_dtype() == torch.float8_e4m3fnuz else 448.0
        self.pins = [torch.empty(self.E * self.NE, dtype=torch.int32, pin_memory=True) for _ in range(8)]
        self.npin = 0

    # ------------------------------------------------------------------------------------------------------------
    def _slot(self):
        self.ring = _ring(self.g_node, self.G, 0)  # current (possibly grown) ring
        self.buf, self.hdl, self.dtab, self.slot_bytes = self.ring.buf, self.ring.hdl, self.ring.dtab, self.ring.slot_bytes
        self.cap = self.slot_bytes // max(self.XRB, self.YRB, self.GRB)
        return self.ring.next()

    def _ensure_cap(self, rows):
        """Grow the shared ring when an exchange needs more than `cap` rows on some GPU of this node."""
        rb = max(self.XRB, self.YRB, self.GRB)
        if rows * rb > _ring(self.g_node, self.G, 0).slot_bytes:
            _ring(self.g_node, self.G, int(rows * 1.25) * rb)
            if dist.get_rank(self.g_node) == 0:
                print(f"A3: symmetric slots grown to {int(rows * 1.25)} rows", flush=True)

    def _barrier(self):
        with _tm("barrier"):
            self.hdl.barrier(channel=0)

    def _rail_rounds(self, send_fn, recv_bufs, tag="rail"):
        """Pairwise rail exchange: round r = 1..NN-1 sends send_fn(r) to node+r and receives into recv_bufs[r] from
        node-r, on stream sr. Returns per-round events (recorded on sr)."""
        NN, main = self.NN, torch.cuda.current_stream()
        evs = [None] * NN
        self.sr.wait_stream(main)
        with torch.cuda.stream(self.sr):
            for r in range(1, NN):
                ops = [dist.P2POp(dist.isend, send_fn(r), self.rail_ranks[(self.node + r) % NN], self.g_rail),
                       dist.P2POp(dist.irecv, recv_bufs[r], self.rail_ranks[(self.node - r) % NN], self.g_rail)]
                with _tm(f"{tag}{r}"):
                    for w in dist.batch_isend_irecv(ops):
                        w.wait()
                evs[r] = torch.cuda.Event()
                evs[r].record(self.sr)
        return evs

    def _src_of_round(self, r):
        """EP rank whose rows arrive in round r (r = 0: my own rows)."""
        return ((self.node - r) % self.NN) * self.G + self.local

    # ------------------------------------------------------------------------------------------------------------
    def plan(self, routing_map):
        with _tm("plan"):
            T, K, NE, E, NEN = routing_map.shape[0], self.K, self.NE, self.E, self.NEN
            dev = routing_map.device
            with _tm("plan_rank"):
                idx = torch.topk(routing_map.to(torch.int8), K, dim=1, sorted=False).indices
                csT = torch.cumsum(routing_map.t().to(torch.int32), 1, dtype=torch.int32)  # [NE, T]
                tt = torch.arange(T, device=dev)[:, None]
                rk = (csT.view(-1)[idx * T + tt] - 1).to(torch.int32)
                cnt = csT[:, -1].contiguous()
            with _tm("plan_ag"):
                allc = torch.empty(E, NE, dtype=torch.int32, device=dev)
                dist.all_gather_into_tensor(allc, cnt, group=self.group)
            pin = self.pins[self.npin]
            self.npin = (self.npin + 1) % len(self.pins)
            pin.copy_(allc.view(-1), non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            # device tables for my node's experts: B[s, e] = offset of expert e in its GPU's x3 + rows from sources < s
            Cn = allc[:, self.node * NEN:(self.node + 1) * NEN].to(torch.int64)
            tot = Cn.sum(0)
            ex = torch.cumsum(tot, 0) - tot
            gbase = ex.view(self.G, self.NLE)[:, :1].expand(self.G, self.NLE).reshape(-1)
            B = ((ex - gbase)[None, :] + torch.cumsum(Cn, 0) - Cn).to(torch.int32).contiguous()
        _dbg("plan done")
        return dict(T=T, idx=idx, rk=rk, B=B, ev=ev, pin=pin, counts=None, posk=[None] * self.NN)

    def _host_counts(self, pl):
        if pl["counts"] is None:
            with _tm("host_wait"):
                pl["ev"].synchronize()
            C = pl.pop("pin").view(self.E, self.NE).clone()
            mine = C[:, self.me * self.NLE:(self.me + 1) * self.NLE].sum(0).to(torch.int64)
            node_tot = C[:, self.node * self.NEN:(self.node + 1) * self.NEN].sum(0).view(self.G, self.NLE).sum(1)
            assert int(node_tot.max()) <= self.cap, f"A3: x3 rows {int(node_tot.max())} > capacity {self.cap}"
            pl["counts"] = C
            pl["tpe"] = mine
            pl["N3"] = int(mine.sum())
        return pl["tpe"], pl["N3"]

    # ------------------------------------------------------------------------------------------------------------
    def _dispatch_fwd(self, x, probs, pl):
        T, H, K, RB, NN = pl["T"], self.H, self.K, self.RB, self.NN
        dev = x.device
        main = torch.cuda.current_stream()
        slot = self._slot()
        with _tm("pack"):
            sbuf = torch.empty(T, RB, dtype=torch.uint8, device=dev)
            NBP = triton.next_power_of_2(max(H // _QB, 1))
            _pack_kernel[(T,)](x, probs, pl["idx"], pl["rk"], sbuf, H, self.NE, x.stride(0), RB, self.s_off,
                               self.p_off, self.c_off, K=K, FP8=self.fp8, NBP=NBP, QB=_QB, FMAX=self.fmax,
                               HP=triton.next_power_of_2(H))
        rbufs = [None] + [torch.empty(T, RB, dtype=torch.uint8, device=dev) for _ in range(1, NN)]
        _dbg("dispatch: packed")
        evs = self._rail_rounds(lambda r: sbuf, rbufs)
        _dbg("dispatch: rail done")
        sp = self.buf[slot:]
        for r in range(NN):
            if r:
                main.wait_event(evs[r])
            rows = sbuf if r == 0 else rbufs[r]
            posk = torch.empty(T, K, dtype=torch.int32, device=dev)
            with _tm(f"push{r}"):
                _push_fwd_kernel[(self.CTAS,)](rows, T, RB, self.p_off, self.c_off, self.NW, pl["B"],
                                               self._src_of_round(r), self.node * self.NEN, self.NLE, sp.data_ptr(), self.dtab,
                                               self.XRB, posk, K=K, NWP=triton.next_power_of_2(self.NW),
                                               NEN=self.NEN, POSBITS=_POSBITS)
            pl["posk"][r] = posk
        if NN > 1:
            for b in rbufs[1:]:
                b.record_stream(self.sr)
        sbuf.record_stream(self.sr)
        _dbg("dispatch: pushes done")
        self._barrier()
        _dbg("dispatch: barrier done")
        tpe, N3 = self._host_counts(pl)
        with _tm("unpack"):
            x3 = torch.empty(N3, H, dtype=x.dtype, device=dev)
            p3 = torch.empty(N3, dtype=torch.float32, device=dev)
            if N3:
                _dequant_kernel[(N3,)](sp, self.XRB, N3, x3, p3, H, self.s_off, self.NW * 4,
                                       NBP=triton.next_power_of_2(max(H // _QB, 1)), QB=_QB, FP8=self.fp8,
                                       HP=triton.next_power_of_2(H))
        pl.update(xdt=x.dtype, pdt=probs.dtype)
        return x3, p3.to(probs.dtype) if probs.dtype != torch.float32 else p3

    def _reduce_path(self, y3, gp3, pl):
        """combine forward (gp3 None) or dispatch backward: Y rows -> pull-sum per round -> rail back -> sum4."""
        T, H, K, NN = pl["T"], self.H, self.K, self.NN
        dev = y3.device
        main = torch.cuda.current_stream()
        slot = self._slot()
        sp = self.buf[slot:]
        N3 = y3.shape[0]
        withgp = gp3 is not None
        with _tm("ycopy"):
            ys = sp[:N3 * self.YRB].view(N3, self.YRB)
            ys[:, :2 * H].view(torch.bfloat16).copy_(y3)
            if withgp:
                ys[:, 2 * H:2 * H + 4].view(torch.float32).copy_(gp3.float()[:, None])
        _dbg("reduce: ycopy done")
        self._barrier()
        _dbg("reduce: barrier done")
        ORB = _pad16(2 * H + (4 * K if withgp else 0))
        allrecv = torch.empty(NN, T, ORB, dtype=torch.uint8, device=dev)
        parts = [allrecv[0]] + [torch.empty(T, ORB, dtype=torch.uint8, device=dev) for _ in range(1, NN)]
        recv = [allrecv[r] for r in range(NN)]
        ev_part = [None] * NN
        # pull-sum the rounds whose partials go back over the rail first (r = 1..NN-1), then my own rows (r = 0)
        order = list(range(1, NN)) + [0]
        for r in order:
            with _tm(f"pull{r}"):
                _pull_sum_kernel[(self.CTAS,)](pl["posk"][r], T, sp.data_ptr(), self.dtab, self.YRB, 2 * H, parts[r], ORB, H,
                                               K=K, HP=triton.next_power_of_2(H), WITH_GP=withgp,
                                               POSBITS=_POSBITS)
            ev_part[r] = torch.cuda.Event()
            ev_part[r].record(main)
        # rail: round r returns the partials of the rows that came from node-r (parts[r]) and receives my tokens'
        # partials from node+r
        NNr = NN
        self.sr.wait_stream(main)
        with torch.cuda.stream(self.sr):
            for r in range(1, NNr):
                self.sr.wait_event(ev_part[r])
                ops = [dist.P2POp(dist.isend, parts[r], self.rail_ranks[(self.node - r) % NN], self.g_rail),
                       dist.P2POp(dist.irecv, recv[r], self.rail_ranks[(self.node + r) % NN], self.g_rail)]
                with _tm(f"rrail{r}"):
                    for w in dist.batch_isend_irecv(ops):
                        w.wait()
        main.wait_stream(self.sr)
        _dbg("reduce: rail done")
        for b in parts[1:] + [allrecv]:
            b.record_stream(self.sr)
        with _tm("sumn"):
            out = torch.empty(T, H, dtype=y3.dtype, device=dev)
            gpk = torch.empty(T, K, dtype=torch.float32, device=dev) if withgp else out
            _sumn_kernel[(T,)](allrecv, T * ORB, NN, ORB, out, gpk, H, K=K, HP=triton.next_power_of_2(H), WITH_GP=withgp)
        return out, (gpk if withgp else None)

    def _spread_path(self, g, pl):
        """combine backward: grad rows g [T, H] -> rail rounds -> push to copy positions -> gx3."""
        T, H, K, NN = pl["T"], self.H, self.K, self.NN
        dev = g.device
        main = torch.cuda.current_stream()
        slot = self._slot()
        sp = self.buf[slot:]
        g = g.contiguous()
        rbufs = [None] + [torch.empty(T, H, dtype=g.dtype, device=dev) for _ in range(1, NN)]
        evs = self._rail_rounds(lambda r: g, rbufs, "brail")
        for r in range(NN):
            if r:
                main.wait_event(evs[r])
            rows = g if r == 0 else rbufs[r]
            with _tm(f"bpush{r}"):
                _push_bwd_kernel[(self.CTAS,)](rows.view(torch.uint8), T, 2 * H, H // 2, pl["posk"][r], sp.data_ptr(), self.dtab, self.GRB, K=K,
                                               NWP=triton.next_power_of_2(H // 2), POSBITS=_POSBITS)
        for b in rbufs[1:]:
            b.record_stream(self.sr)
        g.record_stream(self.sr)
        _dbg("spread: pushes done")
        self._barrier()
        _dbg("spread: barrier done")
        N3 = pl["N3"]
        with _tm("bunpack"):
            gx3 = sp[:N3 * self.GRB].view(N3, self.GRB).view(g.dtype).clone()
        return gx3

    # ------------------------------------------------------------------------------------------------------------
    def dispatch(self, x, probs, pl):
        with _tm("d_total"):
            x3, p3 = _Dispatch.apply(x, probs, self, pl)
        return x3, p3, pl["tpe"]

    def combine(self, y3, pl):
        with _tm("c_total"):
            return _Combine.apply(y3, self, pl)


class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probs, hier, pl):
        ctx.hier, ctx.pl = hier, pl
        return hier._dispatch_fwd(x.contiguous(), probs.contiguous(), pl)

    @staticmethod
    def backward(ctx, gx3, gp3):
        hier, pl = ctx.hier, ctx.pl
        N3 = pl["N3"]
        dev = (gx3 if gx3 is not None else gp3).device
        if gx3 is None:
            gx3 = torch.zeros(N3, hier.H, dtype=pl["xdt"], device=dev)
        if gp3 is None:
            gp3 = torch.zeros(N3, dtype=torch.float32, device=dev)
        with _tm("db_total"):
            gx, gpk = hier._reduce_path(gx3.contiguous(), gp3.contiguous(), pl)
            gprobs = torch.zeros(pl["T"], hier.NE, dtype=pl["pdt"], device=dev).scatter_(1, pl["idx"],
                                                                                         gpk.to(pl["pdt"]))
        return gx, gprobs, None, None


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y3, hier, pl):
        ctx.hier, ctx.pl = hier, pl
        out, _ = hier._reduce_path(y3.contiguous(), None, pl)
        return out

    @staticmethod
    def backward(ctx, g):
        with _tm("cb_total"):
            return ctx.hier._spread_path(g, ctx.pl), None, None
