# bench2 K3 (k3-0003): two-pass cross-entropy for the untied LM head (TP 1), in place of TE's Triton CE.
# TE (transformer_engine/pytorch/triton/cross_entropy.py) makes three passes over the [tokens, vocab] bf16 logits:
# online softmax (read), loss + gradient written into the logits (read + write) in the forward, and an element-wise
# multiply by grad_output (read + write) in the backward: 5 x 3.28 GB per microbatch at mid-4b (vocab 200,064).
# Here the forward only reads the logits (row max and log-sum-exp, loss), and the backward computes
# (softmax - onehot) * grad_output from the logits and the saved lse in one read + write, in place: 3 x 3.28 GB.
# Same maths in fp32; ignore_index -100 gives loss 0 and gradient 0 (as TE). Enabled by BEVERIN_K3_CE=1 in
# megatron/core/extensions/transformer_engine.py (te_parallel_cross_entropy) when the TP group has one rank.
import torch
import triton
import triton.language as tl
import os

_MODE = os.environ.get("BEVERIN_K3_CE_MODE", "chunk")
_NCH = int(os.environ.get("BEVERIN_K3_CE_NCH", "4"))
_FW = int(os.environ.get("BEVERIN_K3_CE_FW", "8"))
_BB = int(os.environ.get("BEVERIN_K3_CE_BB", "4096"))
_BW = int(os.environ.get("BEVERIN_K3_CE_BW", "8"))


@triton.jit
def _ce_fwd_part(x_ptr, pm_ptr, ps_ptr, V, stride, CH, NCH, BLOCK: tl.constexpr):
    # program (row r, chunk j): max and sum(exp(x - max)) over columns [j*CH, (j+1)*CH)
    r = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    base = x_ptr + r * stride
    m = tl.full([BLOCK], float("-inf"), tl.float32)
    s = tl.zeros([BLOCK], tl.float32)
    lo = j * CH
    for v0 in range(lo, lo + CH, BLOCK):
        c = v0 + tl.arange(0, BLOCK)
        x = tl.load(base + c, mask=(c < V) & (c < lo + CH), other=float("-inf")).to(tl.float32)
        mn = tl.maximum(m, x)
        s = s * tl.exp(m - mn) + tl.where(x == float("-inf"), 0.0, tl.exp(x - mn))
        m = tl.where(mn == float("-inf"), m, mn)
    mx = tl.max(m, 0)
    tot = tl.sum(tl.where(m == float("-inf"), 0.0, s * tl.exp(m - mx)), 0)
    tl.store(pm_ptr + r * NCH + j, mx)
    tl.store(ps_ptr + r * NCH + j, tot)


@triton.jit
def _ce_fwd_comb(x_ptr, t_ptr, pm_ptr, ps_ptr, loss_ptr, lse_ptr, stride, NCH, IGN: tl.constexpr, NP: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, NP)
    pm = tl.load(pm_ptr + r * NCH + k, mask=k < NCH, other=float("-inf"))
    ps = tl.load(ps_ptr + r * NCH + k, mask=k < NCH, other=0.0)
    mx = tl.max(pm, 0)
    lse = mx + tl.log(tl.sum(tl.where(pm == float("-inf"), 0.0, ps * tl.exp(pm - mx)), 0))
    t = tl.load(t_ptr + r)
    ok = t != IGN
    xt = tl.load(x_ptr + r * stride + tl.where(ok, t, 0)).to(tl.float32)
    tl.store(lse_ptr + r, lse)
    tl.store(loss_ptr + r, tl.where(ok, lse - xt, 0.0))


@triton.jit
def _ce_fwd(x_ptr, t_ptr, loss_ptr, lse_ptr, V, stride, IGN: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    base = x_ptr + r * stride
    m = tl.full([BLOCK], float("-inf"), tl.float32)
    s = tl.zeros([BLOCK], tl.float32)
    for v0 in range(0, V, BLOCK):
        c = v0 + tl.arange(0, BLOCK)
        x = tl.load(base + c, mask=c < V, other=float("-inf")).to(tl.float32)
        mn = tl.maximum(m, x)
        s = s * tl.exp(m - mn) + tl.where(x == float("-inf"), 0.0, tl.exp(x - mn))
        m = mn
    mx = tl.max(m, 0)
    tot = tl.sum(s * tl.exp(m - mx), 0)
    lse = mx + tl.log(tot)
    t = tl.load(t_ptr + r)
    ok = t != IGN
    xt = tl.load(base + tl.where(ok, t, 0)).to(tl.float32)
    tl.store(lse_ptr + r, lse)
    tl.store(loss_ptr + r, tl.where(ok, lse - xt, 0.0))


@triton.jit
def _ce_bwd(x_ptr, t_ptr, lse_ptr, g_ptr, V, stride, IGN: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    base = x_ptr + r * stride
    t = tl.load(t_ptr + r)
    lse = tl.load(lse_ptr + r)
    g = tl.load(g_ptr + r).to(tl.float32)
    g = tl.where(t != IGN, g, 0.0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mk = c < V
    x = tl.load(base + c, mask=mk, other=0.0).to(tl.float32)
    d = (tl.exp(x - lse) - tl.where(c == t, 1.0, 0.0)) * g
    tl.store(base + c, d.to(x_ptr.dtype.element_ty), mask=mk)


class K3CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, ignore_index=-100):
        shape = logits.shape[:-1]
        V = logits.shape[-1]
        x = logits.reshape(-1, V)
        if x.stride(1) != 1:
            x = x.contiguous()
        t = target.reshape(-1).contiguous()
        n = x.shape[0]
        loss = torch.empty(n, dtype=torch.float32, device=x.device)
        lse = torch.empty(n, dtype=torch.float32, device=x.device)
        if _MODE == "1pass":
            _ce_fwd[(n,)](x, t, loss, lse, V, x.stride(0), IGN=ignore_index, BLOCK=4096, num_warps=8)
        else:
            nch = _NCH
            ch = triton.cdiv(triton.cdiv(V, nch), 2048) * 2048
            pm = torch.empty(n, nch, dtype=torch.float32, device=x.device)
            ps = torch.empty_like(pm)
            _ce_fwd_part[(n, nch)](x, pm, ps, V, x.stride(0), ch, nch, BLOCK=2048, num_warps=_FW)
            _ce_fwd_comb[(n,)](x, t, pm, ps, loss, lse, x.stride(0), nch, IGN=ignore_index,
                               NP=triton.next_power_of_2(nch), num_warps=1)
        ctx.save_for_backward(x, t, lse)
        ctx.ign, ctx.shape = ignore_index, logits.shape
        return loss.view(shape)

    @staticmethod
    def backward(ctx, grad):
        x, t, lse = ctx.saved_tensors
        g = grad.reshape(-1).contiguous()
        _ce_bwd[(x.shape[0], triton.cdiv(x.shape[1], _BB))](x, t, lse, g, x.shape[1], x.stride(0), IGN=ctx.ign,
                                                           BLOCK=_BB, num_warps=_BW)
        return x.view(ctx.shape), None, None


def cross_entropy(logits, target, ignore_index=-100):
    return K3CrossEntropy.apply(logits, target, ignore_index)
