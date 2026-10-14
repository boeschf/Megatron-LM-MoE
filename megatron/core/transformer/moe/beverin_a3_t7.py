# beverin lab2 A3 "T7": round 1's T6 rail stage (one RCCL all-to-all on the 4-rank rail communicator, rows sent once
# per destination node, exact split sizes) with A3's intra-node stage (no RCCL node all-to-all):
#   dispatch: device plan (expert ranks, node positions) -> count all-gather + async D2H -> fused fp8 pack of every
#             (token, destination node) row (quantised once) -> host waits for the counts (event, the pack is
#             already queued) -> rail all-to-all -> push kernel writes each copy straight into the expert-sorted
#             slot of the node-mate that owns the expert (symmetric memory; positions from the all-gathered counts,
#             order = (local expert, source EP rank, token) as Megatron's alltoall dispatcher) -> device barrier ->
#             local dequantisation into x3.
#   combine / dispatch backward: copy rows into my slot -> barrier -> the holder of each rail row pulls and sums its
#             copies from the node-mates (fp32, fixed order) -> rail all-to-all back -> sum over nodes per token.
#   combine backward: grad rows gathered per destination node -> rail all-to-all -> push to the copy positions.
# Host syncs: one event wait per dispatch, after the pack is queued; none in the other three exchanges.
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from megatron.core.transformer.moe import beverin_a3_ep as A
from megatron.core.transformer.moe.beverin_a3_ep import _tm, _dbg, _pad16, _QB, _POSBITS

TIMES = A.TIMES
timing_summary = A.timing_summary


@triton.jit
def _pack_multi_kernel(x_ptr, probs_ptr, idx_ptr, rk_ptr, ppos_ptr, out_ptr, H, NE, x_stride, RB, s_off, p_off,
                       c_off, K: tl.constexpr, NN: tl.constexpr, FP8: tl.constexpr, NBP: tl.constexpr,
                       QB: tl.constexpr, FMAX: tl.constexpr, HP: tl.constexpr):
    # token t -> row [x (fp8 + fp32 scales, or bf16) | probs fp32 K | codes int32 K] written to every sbuf row
    # ppos[t, m] >= 0 (one per destination node m)
    t = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, K)
    e = tl.load(idx_ptr + t * K + k)
    p = tl.load(probs_ptr + t * NE + e).to(tl.float32)
    code = (e.to(tl.int32) << 20) | tl.load(rk_ptr + t * K + k)
    if FP8:
        NB = H // QB
        b = tl.arange(0, NBP)
        c = tl.arange(0, QB)
        bm = b < NB
        cols = b[:, None] * QB + c[None, :]
        msk = bm[:, None] & (c[None, :] < QB)
        x = tl.load(x_ptr + t * x_stride + cols, mask=msk, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=1)
        s = tl.maximum(amax, 1e-20) / FMAX
        q = (x / s[:, None]).to(tl.float8e4b8).to(tl.uint8, bitcast=True)
    else:
        cols1 = tl.arange(0, HP)
        m1 = cols1 < H
        xb = tl.load(x_ptr + t * x_stride + cols1, mask=m1)
    for m in tl.static_range(NN):
        pos = tl.load(ppos_ptr + t * NN + m)
        if pos >= 0:
            row = out_ptr + pos.to(tl.int64) * RB
            if FP8:
                tl.store(row + cols, q, mask=msk)
                tl.store((row + s_off).to(tl.pointer_type(tl.float32)) + b, s, mask=bm)
            else:
                tl.store(row.to(tl.pointer_type(tl.bfloat16)) + cols1, xb, mask=m1)
            tl.store((row + p_off).to(tl.pointer_type(tl.float32)) + k, p)
            tl.store((row + c_off).to(tl.pointer_type(tl.int32)) + k, code)


@triton.jit
def _gsum_kernel(src_ptr, RB, ppos_ptr, out_ptr, gp_ptr, H, K: tl.constexpr, NN: tl.constexpr, HP: tl.constexpr,
                 WITH_GP: tl.constexpr):
    # out[t] = sum over destination nodes m (in order) of src row ppos[t, m] (bf16 rows of RB bytes, fp32 sums);
    # gp[t] = same over the K fp32 values after the row
    t = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, HP)
    cm = cols < H
    kk = tl.arange(0, K)
    a = tl.zeros([HP], dtype=tl.float32)
    g = tl.zeros([K], dtype=tl.float32)
    for m in tl.static_range(NN):
        pos = tl.load(ppos_ptr + t * NN + m)
        if pos >= 0:
            row = src_ptr + pos.to(tl.int64) * RB
            a += tl.load(row.to(tl.pointer_type(tl.bfloat16)) + cols, mask=cm, other=0.0).to(tl.float32)
            if WITH_GP:
                g += tl.load((row + 2 * H).to(tl.pointer_type(tl.float32)) + kk)
    tl.store(out_ptr + t * H + cols, a.to(tl.bfloat16), mask=cm)
    if WITH_GP:
        tl.store(gp_ptr + t * K + kk, g)


class T7EP(A.A3EP):
    """T6-style compact rail all-to-all + A3 intra-node push/pull. Same constructor as A3EP."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.PCTAS = int(os.environ.get("A3_PULL_CTAS", os.environ.get("A3_CTAS", 128)))
        self.CTAS = int(os.environ.get("A3_CTAS", 128))
        self.pins = [torch.empty(self.E * (self.NE + self.NN), dtype=torch.int32, pin_memory=True)
                     for _ in range(8)]

    # ------------------------------------------------------------------------------------------------------------
    def plan(self, routing_map):
        with _tm("plan"):
            T, K, NE, E, NEN, NN = routing_map.shape[0], self.K, self.NE, self.E, self.NEN, self.NN
            dev = routing_map.device
            idx = torch.topk(routing_map.to(torch.int8), K, dim=1, sorted=False).indices
            csT = torch.cumsum(routing_map.t().to(torch.int32), 1, dtype=torch.int32)  # [NE, T]
            tt = torch.arange(T, device=dev)[:, None]
            rk = (csT.view(-1)[idx * T + tt] - 1).to(torch.int32)
            cnt = csT[:, -1]
            need = torch.zeros(T, NN, dtype=torch.int32, device=dev).scatter_(1, idx // NEN, 1)
            csn = torch.cumsum(need, 0, dtype=torch.int32)
            cn = csn[-1]
            vec = torch.cat([cnt, cn])
            allv = torch.empty(E, NE + NN, dtype=torch.int32, device=dev)
            dist.all_gather_into_tensor(allv, vec, group=self.group)
            pin = self.pins[self.npin]
            self.npin = (self.npin + 1) % len(self.pins)
            pin.copy_(allv.view(-1), non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            soff = torch.cumsum(cn, 0, dtype=torch.int32) - cn
            ppos = torch.where(need > 0, soff[None, :] + csn - 1, torch.full_like(csn, -1)).contiguous()
            Cn = allv[:, self.node * NEN:(self.node + 1) * NEN].to(torch.int64)
            tot = Cn.sum(0)
            ex = torch.cumsum(tot, 0) - tot
            gbase = ex.view(self.G, self.NLE)[:, :1].expand(self.G, self.NLE).reshape(-1)
            B = ((ex - gbase)[None, :] + torch.cumsum(Cn, 0) - Cn).to(torch.int32).contiguous()
        return dict(T=T, idx=idx, rk=rk, B=B, ev=ev, pin=pin, ppos=ppos, counts=None)

    def _host_counts(self, pl):
        if pl["counts"] is None:
            with _tm("host_wait"):
                pl["ev"].synchronize()
            V = pl.pop("pin").view(self.E, self.NE + self.NN).clone()
            C, Nn = V[:, :self.NE], V[:, self.NE:]
            mine = C[:, self.me * self.NLE:(self.me + 1) * self.NLE].sum(0).to(torch.int64)
            node_tot = C[:, self.node * self.NEN:(self.node + 1) * self.NEN].sum(0).view(self.G, self.NLE).sum(1)
            self._ensure_cap(int(node_tot.max()))
            rail = [m * self.G + self.local for m in range(self.NN)]
            pl["in1"] = Nn[self.me].tolist()  # rows I send to rail peer m
            pl["out1"] = Nn[rail, self.node].tolist()  # rows rail peer m sends me
            pl["counts"] = True
            pl["tpe"] = mine
            pl["N3"] = int(mine.sum())
        return pl["tpe"], pl["N3"]

    @staticmethod
    def _off(sizes):
        o = [0]
        for v in sizes:
            o.append(o[-1] + v)
        return o

    # ------------------------------------------------------------------------------------------------------------
    def _dispatch_fwd(self, x, probs, pl):
        T, H, K, RB, NN = pl["T"], self.H, self.K, self.RB, self.NN
        dev = x.device
        with _tm("pack"):
            sbuf = torch.empty(NN * T, RB, dtype=torch.uint8, device=dev)
            _pack_multi_kernel[(T,)](x, probs, pl["idx"], pl["rk"], pl["ppos"], sbuf, H, self.NE, x.stride(0), RB,
                                     self.s_off, self.p_off, self.c_off, K=K, NN=NN, FP8=self.fp8,
                                     NBP=triton.next_power_of_2(max(H // _QB, 1)), QB=_QB, FMAX=self.fmax,
                                     HP=triton.next_power_of_2(H))
        tpe, N3 = self._host_counts(pl)
        slot = self._slot()
        sp = self.buf[slot:]
        in1, out1 = pl["in1"], pl["out1"]
        o1 = self._off(out1)
        with _tm("rail"):
            rbuf = torch.empty(o1[-1], RB, dtype=torch.uint8, device=dev)
            dist.all_to_all_single(rbuf, sbuf[:sum(in1)], output_split_sizes=out1, input_split_sizes=in1,
                                   group=self.g_rail)
        del sbuf
        cur, ns = torch.cuda.current_stream(), self.ring.stream
        ns.wait_stream(cur)
        rbuf.record_stream(ns)
        with torch.cuda.stream(ns):
            posk = torch.empty(o1[-1], K, dtype=torch.int32, device=dev)
            with _tm("push"):
                for m in range(NN):
                    R = out1[m]
                    if R:
                        _push_fwd(self, rbuf[o1[m]:], R, m * self.G + self.local, pl["B"], sp, posk[o1[m]:])
            self._barrier()
            with _tm("unpack"):
                x3 = torch.empty(N3, H, dtype=x.dtype, device=dev)
                p3 = torch.empty(N3, dtype=torch.float32, device=dev)
                if N3:
                    A._dequant_kernel[(N3,)](sp, self.XRB, N3, x3, p3, H, self.s_off, self.NW * 4,
                                             NBP=triton.next_power_of_2(max(H // _QB, 1)), QB=_QB, FP8=self.fp8,
                                             HP=triton.next_power_of_2(H))
        del rbuf
        cur.wait_stream(ns)
        for t in (x3, p3, posk):
            t.record_stream(cur)
        pl.update(posk=posk, o1=o1, xdt=x.dtype, pdt=probs.dtype)
        return x3, p3

    def _reduce_path(self, y3, gp3, pl):
        T, H, K, NN = pl["T"], self.H, self.K, self.NN
        dev = y3.device
        slot = self._slot()
        sp = self.buf[slot:]
        N3 = y3.shape[0]
        withgp = gp3 is not None
        ORB = _pad16(2 * H + (4 * K if withgp else 0))
        o1, in1, out1 = pl["o1"], pl["in1"], pl["out1"]
        R1 = o1[-1]
        cur, ns = torch.cuda.current_stream(), self.ring.stream
        ns.wait_stream(cur)
        y3.record_stream(ns)
        if withgp:
            gp3.record_stream(ns)
        with torch.cuda.stream(ns):
            with _tm("ycopy"):
                ys = sp[:N3 * self.YRB].view(N3, self.YRB)
                ys[:, :2 * H].view(torch.bfloat16).copy_(y3)
                if withgp:
                    ys[:, 2 * H:2 * H + 4].view(torch.float32).copy_(gp3.float()[:, None])
            self._barrier()
            with _tm("pull"):
                b1 = torch.empty(R1, ORB, dtype=torch.uint8, device=dev)
                if R1:
                    A._pull_sum_kernel[(self.PCTAS,)](pl["posk"], R1, sp.data_ptr(), self.dtab, self.YRB, 2 * H, b1,
                                                      ORB, H, K=K, HP=triton.next_power_of_2(H), WITH_GP=withgp,
                                                      POSBITS=_POSBITS)
        cur.wait_stream(ns)
        b1.record_stream(cur)
        with _tm("rrail"):
            r = torch.empty(sum(in1), ORB, dtype=torch.uint8, device=dev)
            dist.all_to_all_single(r, b1, output_split_sizes=in1, input_split_sizes=out1, group=self.g_rail)
        del b1
        with _tm("gsum"):
            out = torch.empty(T, H, dtype=y3.dtype, device=dev)
            gpk = torch.empty(T, K, dtype=torch.float32, device=dev) if withgp else out
            _gsum_kernel[(T,)](r, ORB, pl["ppos"], out, gpk, H, K=K, NN=NN, HP=triton.next_power_of_2(H),
                               WITH_GP=withgp)
        return out, (gpk if withgp else None)

    def _spread_path(self, g, pl):
        T, H, K, NN = pl["T"], self.H, self.K, self.NN
        dev = g.device
        slot = self._slot()
        sp = self.buf[slot:]
        g = g.contiguous()
        in1, out1, o1 = pl["in1"], pl["out1"], pl["o1"]
        with _tm("bpack"):
            if "gidx" not in pl:
                pp = pl["ppos"]
                valid = pp >= 0
                gidx = torch.empty(sum(in1), dtype=torch.long, device=dev)
                gidx[pp[valid].long()] = torch.arange(T, device=dev)[:, None].expand(T, NN)[valid]
                pl["gidx"] = gidx
            sb = g.index_select(0, pl["gidx"])
        with _tm("brail"):
            rb = torch.empty(o1[-1], H, dtype=g.dtype, device=dev)
            dist.all_to_all_single(rb, sb, output_split_sizes=out1, input_split_sizes=in1, group=self.g_rail)
        del sb
        cur, ns = torch.cuda.current_stream(), self.ring.stream
        ns.wait_stream(cur)
        rb.record_stream(ns)
        N3 = pl["N3"]
        with torch.cuda.stream(ns):
            with _tm("bpush"):
                R1 = o1[-1]
                if R1:
                    A._push_bwd_kernel[(self.CTAS,)](rb.view(torch.uint8), R1, 2 * H, H // 2, pl["posk"],
                                                     sp.data_ptr(), self.dtab, self.GRB, K=K,
                                                     NWP=triton.next_power_of_2(H // 2), POSBITS=_POSBITS)
            self._barrier()
            with _tm("bunpack"):
                gx3 = sp[:N3 * self.GRB].view(N3, self.GRB).view(g.dtype).clone()
        del rb
        cur.wait_stream(ns)
        gx3.record_stream(cur)
        return gx3


def _push_fwd(h, rows, R, s, B, sp, posk):
    A._push_fwd_kernel[(h.CTAS,)](rows, R, h.RB, h.p_off, h.c_off, h.NW, B, s, h.node * h.NEN, h.NLE,
                                  sp.data_ptr(), h.dtab, h.XRB, posk, K=h.K, NWP=triton.next_power_of_2(h.NW),
                                  NEN=h.NEN, POSBITS=_POSBITS)
