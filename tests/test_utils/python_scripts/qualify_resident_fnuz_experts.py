"""Standalone resident expert qualification; no distributed launcher or test assets.

From the frozen Megatron checkout with image910 Python on one gfx942:
  python tests/test_utils/python_scripts/qualify_resident_fnuz_experts.py \
      --trace-dir /new/rank-owned/path

Use --cpu for lifecycle/oracle checks with CPU PyTorch and no TE or Triton.
The native probe requires an image PyTorch with functional HIP kernel tracing.
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F


WIDTH, FFN, BLOCK = 1792, 1920, 128
SOURCE_ROOT = Path(__file__).resolve().parents[3]
RAGGED = (0, 1, 0, 127, 128, 0, 129, 0, 0, 0, 0, 255, 0, 0, 0, 3)
SKEWED = (0,) * 15 + (1025,)
EMPTY = (0,) * 16
OPERATIONS = {
    "fc1_fwd": ("TN", torch.bfloat16),
    "fc2_fwd": ("TN", torch.bfloat16),
    "fc2_dgrad": ("NN", torch.bfloat16),
    "fc1_dgrad": ("NN", torch.bfloat16),
    "fc2_wgrad": ("NT", torch.float32),
    "fc1_wgrad": ("NT", torch.float32),
}
# (\|error\|2 / \|reference\|2, max error / max |reference|). These are matching-
# recipe bounds, not FN parity bounds. Full-chain includes quantization of
# intermediate results; FP32 accumulation is also tested directly below.
LIMITS = {"y": (.018, .04), "dx": (.018, .04), "dp": (.018, .04), "dw": (.018, .04)}


def error_norms(actual, expected):
    assert actual.shape == expected.shape
    if actual.numel() == 0:
        return (0.0, 0.0)
    actual, expected = actual.float(), expected.float()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    diff = actual - expected
    return (
        (diff.norm() / expected.norm().clamp_min(1e-30)).item(),
        (diff.abs().max() / expected.abs().max().clamp_min(1e-30)).item(),
    )


def check_error(label, actual, expected, limits):
    norms = error_norms(actual, expected)
    assert all(n <= t for n, t in zip(norms, limits)), (label, norms, limits)
    return norms


def quant_blocks(value, max_fp8, dtype, axes):
    """Independent finite cast oracle (FP32 inverse scale, RNE, saturating cast)."""
    m, k = value.shape
    m_block, k_block = axes
    tiles = value.float().reshape(m // m_block, m_block, k // k_block, k_block)
    amax = tiles.abs().amax(dim=(1, 3), keepdim=True)
    if dtype == torch.float8_e4m3fn:
        # GH200 scale convention for the diagnostic comparison only; this
        # does not emulate DeepGEMM reduction order or Triton code generation.
        inverse = (amax / max_fp8).clamp_min(1e-30)
        if axes != (BLOCK, 1):
            inverse = torch.where(amax == 0, 1.0, inverse)
        scale = inverse.reciprocal()
    else:
        scale = torch.where(
            amax == 0, 1.0, (max_fp8 / amax).clamp_max(torch.finfo(torch.float32).max)
        )
        inverse = scale.reciprocal()
    payload = (tiles * scale).clamp(-max_fp8, max_fp8).to(dtype)
    return payload.reshape(m, k), inverse.reshape(m // m_block, k // k_block)


def qdq(value, fmt, axes):
    q, inverse = quant_blocks(value, *fmt, axes)
    mm, kk = axes
    return (
        q.float().view(q.shape[0] // mm, mm, q.shape[1] // kk, kk)
        * inverse[:, None, :, None]
    ).reshape_as(value)


def sssglu(a, probs):
    gate, up = a.float().chunk(2, dim=-1)
    z = gate - 1.
    return ((.5 + z / (1. + z.abs())) * up * probs).to(a.dtype)


def dsssglu(ds, a, probs):
    gate, up = a.float().chunk(2, dim=-1)
    z = gate - 1.
    denom = 1. + z.abs()
    g = .5 + z / denom
    d = ds.float()
    return (
        torch.cat((d * probs / denom.square() * up, d * probs * g), dim=-1).to(a.dtype),
        (d * g * up).sum(-1),
    )


@torch.no_grad()
def oracle_one(x, prob, dy, w1, w2, fmt):
    """Reference equations, with independently quantized rows, weights and columns."""
    m = x.shape[0]
    pad = (-m) % BLOCK
    x, dy = F.pad(x, (0, 0, 0, pad)), F.pad(dy, (0, 0, 0, pad))
    prob = F.pad(prob, (0, pad))[:, None]
    xr, dyr = (qdq(v, fmt, (1, BLOCK)) for v in (x, dy))
    w1q, w2q = (qdq(w, fmt, (BLOCK, BLOCK)) for w in (w1, w2))
    a = (xr @ w1q.T).to(torch.bfloat16)
    y = (qdq(sssglu(a, prob), fmt, (1, BLOCK)) @ w2q.T).to(torch.bfloat16)
    ab = qdq(a, fmt, (1, BLOCK)).to(torch.bfloat16)
    ds = (dyr @ w2q).to(torch.bfloat16)
    da, dp = dsssglu(ds, ab, prob)
    dx = (qdq(da, fmt, (1, BLOCK)) @ w1q).to(torch.bfloat16)
    sb, xb = sssglu(ab, prob), xr.to(torch.bfloat16)
    a_t, y_t, s_t, x_t = (qdq(v, fmt, (BLOCK, 1)).T for v in (da, dy, sb, xb))
    return y[:m], dx[:m], dp[:m], a_t @ x_t.T, y_t @ s_t.T


def load_adapter():
    # Import the implementation without Megatron's optional package tree so
    # the same CPU check can run on developer machines with CPU PyTorch only.
    path = SOURCE_ROOT / "megatron/core/transformer/moe/resident_fp8.py"
    name = "_qualify_resident_fnuz_adapter"
    spec = importlib.util.spec_from_file_location(name, path)
    module = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ObservedBackend:
    def __init__(self, backend):
        self.backend = backend
        self.calls = Counter()
        self.casts = Counter()

    def __getattr__(self, name):
        fn = getattr(self.backend, name)
        if name not in ("weight", "row", "column"):
            return fn

        def cast(value):
            self.casts[name] += 1
            return fn(value)

        return cast

    def gemm(self, operation, a, b, out, *, layout, accumulate=False):
        if out:
            self.calls[operation] += len(out)
            assert (layout, out[0].dtype) == OPERATIONS[operation]
            assert accumulate == operation.endswith("_wgrad")
        return self.backend.gemm(operation, a, b, out, layout=layout, accumulate=accumulate)


def make_raw(adapter, device, n_experts, width, ffn):
    generator = torch.Generator().manual_seed(42158)
    params = [
        [
            torch.nn.Parameter(
                (torch.randn(shape, generator=generator) * .02).to(device, torch.bfloat16)
            )
            for _ in range(n_experts)
        ]
        for shape in ((2 * ffn, width), (width, ffn))
    ]
    return adapter.ResidentFP8Experts(*params)


def samples(splits, width, device, seed):
    g = torch.Generator().manual_seed(seed)
    shape = (sum(splits), width)
    x = torch.randn(shape, generator=g).to(device, torch.bfloat16)
    dy = (torch.randn(shape, generator=g) * .25).to(device, torch.bfloat16)
    probs = (torch.rand(shape[0], generator=g) * .5).to(device)
    x.requires_grad_()
    probs.requires_grad_()
    return x, probs, dy


class CPUBackend:
    """Emulate only device operations; the adapter's real queues/autograd are exercised."""

    fmt = (240., torch.float8_e4m3fnuz)

    def weight(self, x):
        return qdq(x, self.fmt, (BLOCK, BLOCK))

    def row(self, x):
        return qdq(x, self.fmt, (1, BLOCK))

    def column(self, x):
        return qdq(x, self.fmt, (BLOCK, 1))

    activation = staticmethod(sssglu)
    activation_backward = staticmethod(dsssglu)

    @staticmethod
    def decode_row(value):
        return value.to(torch.bfloat16)

    @staticmethod
    def keep_on(value, stream):
        pass

    @staticmethod
    def gemm(operation, a, b, out, *, layout, accumulate):
        for aa, bb, dest in zip(a, b, out):
            val = {"TN": lambda: bb @ aa.T, "NN": lambda: bb @ aa, "NT": lambda: bb.T @ aa}[layout]()
            if accumulate:
                dest.add_(val)
            else:
                dest.copy_(val)


def expect_error(fn, substr):
    try:
        fn()
    except RuntimeError as err:
        assert substr in str(err), err
    else:
        raise AssertionError(f"Expected error matching {substr!r}")


class CPUStream:
    def __init__(self):
        self.waits = []

    def wait_event(self, event):
        self.waits.append(event)


class CPUEvent:
    def record(self, stream):
        self.stream = stream


def check_list_gemm_contract(adapter):
    """Check the image 910 Python/pybind list contract without executing TE."""
    backend = object.__new__(adapter._NativeBackend)
    a = [torch.empty(128, 128), torch.empty(128, 128)]
    b = [torch.empty(128, 128), torch.empty(256, 128)]
    out = [torch.empty(128, 128), torch.empty(128, 128)]
    calls = []

    def grouped(A, B, D, quantizers, *, m_splits, **options):
        assert A is a and B is b and D is out and quantizers == [None, None]
        assert m_splits == [128, 256] and all(type(n) is int for n in m_splits)
        assert options["single_output"] is False
        calls.append((options["layout"], options["grad"], options["accumulate"]))

    backend._grouped = grouped
    for op, (layout, _) in OPERATIONS.items():
        backend.gemm(op, a, b, out, layout=layout, accumulate=(layout == "NT"))
    assert calls == [("TN", False, False)] * 2 + [("NN", True, False)] * 2 + [("NT", True, True)] * 2


def cpu_check():
    torch.set_num_threads(2)
    adapter = load_adapter()
    check_list_gemm_contract(adapter)
    experts = make_raw(adapter, "cpu", 3, 128, 128)
    backend = experts._backend = ObservedBackend(CPUBackend())
    producer, consumer, consumer2 = CPUStream(), CPUStream(), CPUStream()
    notified = []
    for p in experts.params:
        p.main_grad = torch.full(p.shape, .125, dtype=torch.float32)
        experts.register_main_grad_ready(p, notified.append)
    expect_error(experts.drain, "no completed backward")
    with (
        patch.object(torch.cuda, "current_stream", return_value=producer) as current_stream,
        patch.object(torch.cuda, "Event", CPUEvent),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        patch.object(torch.Tensor, "record_stream"),
    ):
        cases = ((1, 0, 129), (0, 0, 0))
        products = []
        for i, splits in enumerate(cases):
            x, p, dy = samples(splits, 128, "cpu", 10 + i)
            y = experts(x, splits, p)
            products.append((x, p, y, dy))
        expect_error(experts.clear_weight_cache, "undrained")
        for x, p, y, dy in products:
            y.backward(dy)
            assert x.grad.shape == x.shape and p.grad.shape == p.shape
        x, p, y, dy = products[0]
        references = []
        for i, sl in ((0, slice(0, 1)), (2, slice(1, 130))):
            yr, dx, dp, dw1, dw2 = oracle_one(
                x[sl], p[sl], dy[sl], experts.fc1[i], experts.fc2[i], CPUBackend.fmt
            )
            references.extend(((experts.fc1[i], dw1), (experts.fc2[i], dw2)))
            for actual, expected in ((y[sl], yr), (x.grad[sl], dx), (p.grad[sl], dp)):
                torch.testing.assert_close(actual, expected)
        first, second = experts._pending
        assert first.active == (0, 2) and second.active == ()
        assert first.ready.stream is producer and second.ready.stream is producer
        assert not notified and len(experts._weights) == 2
        current_stream.return_value = consumer
        ready_destination = experts.fc1[0]
        destination = ready_destination.main_grad
        ready_destination.main_grad = destination.to(torch.bfloat16)
        expect_error(experts.drain, "FP32 main_grad only")
        assert len(experts._pending) == 2 and not notified
        ready_destination.main_grad = destination
        experts.drain()
        assert consumer.waits == [first.ready]
        assert [id(p) for p in notified] == [id(p) for p in experts.fc2 + experts.fc1]
        assert all(p.grad is None for p in experts.params)
        for p in (experts.fc1[1], experts.fc2[1]):
            assert torch.all(p.main_grad == .125)
        for p, contribution in references:
            torch.testing.assert_close(p.main_grad, contribution + .125)
        snapshots = [p.main_grad.clone() for p in experts.params]
        last = experts._last_drain
        current_stream.return_value = consumer2
        experts.drain()
        assert consumer2.waits == [second.ready, last]
        assert len(notified) == 2 * len(experts.params)
        assert all(torch.equal(p.main_grad, v) for p, v in zip(experts.params, snapshots))
        assert experts._outstanding == 0 and set(backend.calls.values()) == {2}
        assert backend.casts == {"weight": 4, "row": 10, "column": 8}
        expect_error(experts.drain, "no completed backward")

        x, p, _, _ = products[0]
        with torch.no_grad():
            y = experts(x, cases[0], p)
            assert backend.casts["weight"] == 4
            param = experts.fc1[0]
            param.data.mul_(-1.)
            torch.autograd.graph.increment_version(param)
            refreshed = experts(x, cases[0], p)
            assert backend.casts["weight"] == 8 and not torch.equal(y, refreshed)
        y = experts(x, cases[0], p)
        torch.autograd.graph.increment_version(experts.fc2[0])
        expect_error(lambda: y.backward(torch.ones_like(y)), "changed between forward and backward")
    print("resident FNUZ CPU lifecycle: PASS (software GEMMs and simulated streams)")


def codec_check(backend, device):
    # Exact FNUZ maximum, signed zeros, subnormal RNE ties, normal ties,
    # alternate signed rows, and independent FP32 scale words. All shapes are
    # actual width. Descriptive large/tiny finite probes below use TE policy.
    v = torch.zeros((256, WIDTH), device=device, dtype=torch.bfloat16)
    values = (240., -240., -0., 2**-10, 3 * 2**-11, 5 * 2**-11, 1.0625, 1.1875)
    expected = (127, 255, 0, 1, 2, 2, 64, 66)
    v[0, : len(values)] = torch.tensor(values, device=device, dtype=v.dtype)
    v[0, BLOCK : BLOCK + len(values)] = v[0, : len(values)] * .5
    v[1, :BLOCK] = -v[0, :BLOCK]
    v[2, :BLOCK] = torch.finfo(torch.bfloat16).tiny
    v[3, :BLOCK] = 1.e20
    got = backend.row(v)
    assert got._rowwise_scale_inv.dtype == torch.float32
    assert got._rowwise_data[0, :len(values)].tolist() == list(expected)
    assert got._rowwise_data[0, BLOCK:BLOCK + len(values)].tolist() == list(expected)
    assert got._rowwise_data[4:].count_nonzero() == 0
    assert got._rowwise_scale_inv[:2, 0].tolist() == [1., .5]
    assert got._rowwise_scale_inv[2:, 4:].unique().tolist() == [1.]
    assert torch.isfinite(got._rowwise_scale_inv).all()
    assert torch.isfinite(backend.decode_row(got)).all()
    print("native FNUZ bytes, RNE midpoint, zero-scale and finite extreme fixtures: PASS")


def assert_native_names(names, count, output_type, operation):
    """Require actual device dispatch, not Python/C++ call names or CPU markers."""
    kernels = [n for n in names if "cdna3::blockwise_fp8::micro_tk<" in n]
    assert len(kernels) == count, (operation, count, names)
    for kernel in kernels:
        assert re.search(
            r"micro_tk<\s*__hip_fp8_e4m3_fnuz,\s*__hip_fp8_e4m3_fnuz,\s*"
            + output_type + r"\s*,", kernel
        ), (operation, kernel)
    extra = [n for n in names if n not in kernels and re.search(r"gemm|matmul", n, re.I)]
    assert not extra, (operation, extra)


def prove_native_dispatch(backend, device, trace_dir):
    trace_dir.mkdir(parents=True, exist_ok=False)
    # Distinct ragged padded groups: work must reach a native kernel for both.
    for operation, (layout, dtype) in OPERATIONS.items():
        fc1 = operation.startswith("fc1")
        out_width, in_width = ((2 * FFN, WIDTH) if fc1 else (WIDTH, FFN))
        a, b, out, expected = [], [], [], []
        for m in (128, 256):
            # Exact, distinct integer products exercise beta=1 and an FP32
            # initial value which cannot round-trip through BF16.
            bf16 = dict(device=device, dtype=torch.bfloat16)
            if layout == "NT":
                a.append(backend.column(torch.ones((m, in_width), **bf16)))
                b.append(backend.column(torch.ones((m, out_width), **bf16)))
                dest = torch.full((out_width, in_width), .2509765625, device=device, dtype=dtype)
                expected.append(m + .2509765625)
            elif layout == "TN":
                a.append(backend.weight(torch.ones((out_width, in_width), **bf16)))
                b.append(backend.row(torch.ones((m, in_width), **bf16)))
                dest = torch.empty((m, out_width), device=device, dtype=dtype)
                expected.append(in_width)
            else:
                a.append(backend.weight(torch.ones((out_width, in_width), **bf16)))
                b.append(backend.row(torch.ones((m, out_width), **bf16)))
                dest = torch.empty((m, in_width), device=device, dtype=dtype)
                expected.append(out_width)
            out.append(dest)
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            backend.gemm(operation, a, b, out, layout=layout, accumulate=(layout == "NT"))
            torch.cuda.synchronize(device)
        prof.export_chrome_trace(str(trace_dir / f"{operation}.json"))
        names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        assert_native_names(names, 2, "float" if dtype == torch.float32 else "__hip_bfloat16",
                            operation)
        # 240 reciprocal isn't exact; bound FP32 multiply/accumulate error
        # without permitting a BF16-rounded wgrad or a missing beta term.
        for actual, value in zip(out, expected):
            if dtype == torch.float32:
                torch.testing.assert_close(actual, torch.full_like(actual, value), rtol=0, atol=1e-4)
            else:
                assert torch.all(actual == value)
        print(f"{operation}: 2 native FNUZ kernel launches, exact fixture passed")


def native_check(device, trace_dir):
    from megatron.core.transformer.moe.resident_fp8 import _NativeBackend

    module = make_te_experts(device, selected=True)
    baseline = make_te_experts(device, selected=False)
    assert list(module.state_dict()) == list(baseline.state_dict())
    for (name, param), (ref_name, ref_param) in zip(
        module.named_parameters(), baseline.named_parameters(), strict=True
    ):
        assert name == ref_name and torch.equal(param, ref_param)
        assert not getattr(ref_param, "resident_fp8_delayed_wgrad", False)
    del baseline

    experts = module._resident_fp8
    native = _NativeBackend(device)
    backend = experts._backend = ObservedBackend(native)
    codec_check(native, device)
    prove_native_dispatch(native, device, trace_dir)
    base = torch.cuda.current_stream(device)
    producer = torch.cuda.Stream(device=device)
    consumers = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
    fmt = (240., torch.float8_e4m3fnuz)
    fn = (448., torch.float8_e4m3fn)
    ready = []
    expected = {}
    observed_errors = {}
    # Observe each write asynchronously from the callback's own stream.
    snapshots = []
    ids = [id(p) for p in module.parameters()]
    for p in experts.params:
        p.main_grad = torch.full_like(p, .0313720703125, dtype=torch.float32)
        expected[p] = p.main_grad.clone()

        def on_ready(param):
            assert param.grad is None
            ready.append((id(param), torch.cuda.current_stream(device)))
            snapshots.append(param.main_grad.clone())

        module.register_resident_fp8_main_grad_ready(p, on_ready)
    with torch.no_grad():
        ckpt = {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in module.state_dict().items()}
    products = []
    # Two inflight forwards, their backwards in reverse order, then empty last.
    cases = [RAGGED, SKEWED, EMPTY]
    for j, splits in enumerate(cases):
        x, p, dy = samples(splits, WIDTH, device, 597 + j)
        divisions = torch.tensor(splits)
        producer.wait_stream(base)
        with torch.cuda.stream(producer):
            y, bias = module(x, divisions, p)
            assert bias is None and y.dtype == torch.bfloat16
        products.append((splits, x, p, dy, y))
    assert backend.casts["weight"] == 12  # six active experts, each cached once
    fn_deltas = []
    n_active = sum(bool(n) for s in cases for n in s)
    assert not ready and backend.calls == {"fc1_fwd": n_active, "fc2_fwd": n_active}
    for splits, x, p, dy, y in (products[1], products[0], products[2]):
        with torch.cuda.stream(producer):
            y.backward(dy)
    assert not ready and len(experts._pending) == 3
    assert backend.casts == {"weight": 12, "row": 5 * n_active, "column": 4 * n_active}
    base.wait_stream(producer)
    for case, (splits, x, p, dy, y) in enumerate((products[1], products[0], products[2])):
        start = 0
        for i, n in enumerate(splits):
            if n:
                selected = slice(start, start + n)
                args = x[selected], p[selected], dy[selected], experts.fc1[i], experts.fc2[i]
                yr, dx, dp, dw1, dw2 = oracle_one(*args, fmt)
                fn_y, fn_dx, fn_dp, fn_dw1, fn_dw2 = oracle_one(*args, fn)
                for name, actual, ref, cross in (
                    ("y", y[selected], yr, fn_y), ("dx", x.grad[selected], dx, fn_dx),
                    ("dp", p.grad[selected], dp, fn_dp),
                ):
                    error = check_error(f"case {case} expert {i} {name}", actual, ref, LIMITS[name])
                    observed_errors.setdefault(name, []).append(error)
                    fn_deltas.append((name, check_error(name, ref, cross, (float("inf"),) * 2)))
                for param, wgrad, cross in (
                    (experts.fc1[i], dw1, fn_dw1), (experts.fc2[i], dw2, fn_dw2)
                ):
                    expected[param].add_(wgrad)
                    fn_deltas.append(("dw", check_error("dw", wgrad, cross, (float("inf"),) * 2)))
            start += n
        if not sum(splits):
            assert y.numel() == x.grad.numel() == p.grad.numel() == 0
        assert p.grad.dtype == torch.float32
        consumer = consumers[case % 2]
        # The consumer has not otherwise waited on producer; the adapter owns
        # that dependency. It also has to chain the two different drain streams.
        with torch.cuda.stream(consumer):
            module.backward_dw()
        base.wait_stream(consumer)
        order = experts.fc2 + experts.fc1
        num = len(order)
        assert ready[-num:] == [(id(param), consumer) for param in order]
        for param, snapshot in zip(order, snapshots[-num:]):
            error = check_error(
                "FP32 main grad at callback",
                snapshot - .0313720703125,
                expected[param] - .0313720703125,
                LIMITS["dw"],
            )
            observed_errors.setdefault("dw", []).append(error)
            if case == 2:
                previous = snapshots[-2 * num : -num]
                assert torch.equal(snapshot, previous[order.index(param)])
    assert len(ready) == 3 * len(experts.params) and experts._outstanding == 0
    assert backend.calls == {name: n_active for name in OPERATIONS}
    expect_error(module.backward_dw, "no completed backward")

    # Simulate optimizer p.data write + explicit version seal and then restore
    # through the stock model API. No derived state or parameter IDs may change.
    splits, x, p, _, _ = products[0]
    with torch.no_grad():
        original = module(x, torch.tensor(splits), p)[0]
        target = experts.fc1[1]
        target.data.mul_(-1.)
        torch.autograd.graph.increment_version(target)
        updated = module(x, torch.tensor(splits), p)[0]
        assert not torch.equal(original, updated)
        module.load_state_dict(ckpt)
        restored = module(x, torch.tensor(splits), p)[0]
        assert torch.equal(original, restored)
        assert ids == [id(param) for param in module.parameters()]
        assert list(ckpt) == list(module.state_dict())
    print("Matching FNUZ oracle worst (relative L2, scaled max):",
          {name: tuple(map(max, zip(*errors))) for name, errors in observed_errors.items()})
    print("FN-format oracle versus FNUZ oracle, measured only:", fn_deltas)
    print("resident FNUZ GPU native dispatch, FNUZ arithmetic and lifecycle: PASS")


@torch.no_grad()
def diagnose_skewed_fc2(device):
    """Localize job 756614's mismatch without changing the acceptance test."""
    from megatron.core.transformer.moe.resident_fp8 import _NativeBackend

    module = make_te_experts(device, selected=True)
    native = _NativeBackend(device)
    x, p, _ = samples(SKEWED, WIDTH, device, 598)
    m = x.shape[0]
    padding = (-m) % BLOCK
    xp = F.pad(x, (0, 0, 0, padding))
    pp = F.pad(p, (0, padding))[:, None]
    expert = module._resident_fp8
    w1, w2 = expert.fc1[15], expert.fc2[15]
    fmt = (240., torch.float8_e4m3fnuz)
    row, square = (1, BLOCK), (BLOCK, BLOCK)

    def compare(a, b):
        return {"error_norms": error_norms(a, b), "unequal_values": int((a != b).sum())}

    def gemm(op, weight, operand, width):
        out = xp.new_empty((m + padding, width))
        native.gemm(op, [weight], [operand], [out], layout="TN")
        return out

    def cast_report(native_q, original, axes):
        payload, scale = quant_blocks(original, *fmt, axes)
        # The TE storage carries typed native FNUZ plus the inverse scales.
        # Compare the raw codes, and compare FP32 decoded values independently
        # of the physical scale buffer's padding or layout.
        actual = native_q._rowwise_data
        assert actual.dtype == torch.uint8 and actual.shape == payload.shape
        return {
            "bit_mismatches": int((actual != payload.view(torch.uint8)).sum()),
            "dequant": compare(native_q.dequantize(dtype=torch.float32), qdq(original, fmt, axes)),
            "logical_scale_shape": list(scale.shape),
        }

    nx, nw1, nw2 = native.row(xp), native.weight(w1), native.weight(w2)
    native_a = gemm("fc1_fwd", nw1, nx, 2 * FFN)
    original_a = (qdq(xp, fmt, row) @ qdq(w1, fmt, square).T).bfloat16()
    conditioned_a = (
        nx.dequantize(dtype=torch.float32) @ nw1.dequantize(dtype=torch.float32).T
    ).bfloat16()
    native_a[m:].zero_()
    native_s = native.activation(native_a, pp)
    native_s[m:].zero_()
    local_s = sssglu(native_a, pp)
    original_s = sssglu(original_a, pp)
    ns = native.row(native_s)
    native_y = gemm("fc2_fwd", nw2, ns, WIDTH)
    conditioned_y = (
        ns.dequantize(dtype=torch.float32) @ nw2.dequantize(dtype=torch.float32).T
    ).bfloat16()
    original_y = (qdq(original_s, fmt, row) @ qdq(w2, fmt, square).T).bfloat16()
    isolated_s_cast_y = (qdq(native_s, fmt, row) @ qdq(w2, fmt, square).T).bfloat16()
    same_a_y = (qdq(local_s, fmt, row) @ qdq(w2, fmt, square).T).bfloat16()
    combined_y, bias = module(x, torch.tensor(SKEWED), p)
    assert bias is None
    report = {
        "case": "skewed expert 15",
        "rows": m,
        "cast_input": cast_report(nx, xp, row),
        "cast_fc1_weight": cast_report(nw1, w1, square),
        "cast_fc2_weight": cast_report(nw2, w2, square),
        "fc1_vs_original": compare(native_a[:m], original_a[:m]),
        "fc1_given_native_payloads": compare(native_a[:m], conditioned_a[:m]),
        "sssglu_given_native_fc1": compare(native_s[:m], local_s[:m]),
        "cast_fc2_activation": cast_report(ns, native_s, row),
        "fc2_given_native_payloads": compare(native_y[:m], conditioned_y[:m]),
        "fc2_given_native_activation": compare(native_y[:m], isolated_s_cast_y[:m]),
        "fc2_with_torch_sssglu_given_native_fc1": compare(same_a_y[:m], original_y[:m]),
        "fc2_vs_original": compare(native_y[:m], original_y[:m]),
        "adapter_vs_explicit_native_chain": compare(combined_y, native_y[:m]),
        "unchanged_acceptance_limits": LIMITS["y"],
        "verdict": "diagnostic only",
    }
    print("FNUZ_EXPERT_ORACLE_DIAGNOSTIC", json.dumps(report, sort_keys=True), flush=True)


def make_te_experts(device, *, selected):
    from types import SimpleNamespace

    from megatron.core.activations import sssglu_act
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear, TERowParallelGroupedLinear,
    )
    from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.manual_seed(42158)
    config = TransformerConfig(
        num_layers=1, hidden_size=3584, num_attention_heads=28,
        moe_ffn_hidden_size=FFN, moe_latent_size=WIDTH,
        num_moe_experts=16, add_bias_linear=False, gated_linear_unit=True,
        activation_func=sssglu_act, params_dtype=torch.bfloat16, bf16=True,
        gradient_accumulation_fusion=False, moe_grouped_gemm=True,
    )
    # Exercise the selected TEGroupedMLP independently of parent-owned selector
    # validation and the distributed schedule; no alternate expert class.
    config.delay_wgrad_compute = True
    config.moe_resident_fp8_fnuz = selected
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    module = TEGroupedMLP(
        16, config, GroupedMLPSubmodules(TEColumnParallelGroupedLinear, TERowParallelGroupedLinear),
        SimpleNamespace(ep=group, expt_tp=group),
    )
    assert all(p.device == device for p in module.parameters())
    assert all(
        getattr(p, "resident_fp8_delayed_wgrad", False) == selected for p in module.parameters()
    )
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu", action="store_true")
    mode.add_argument("--diagnose-skewed-fc2", action="store_true")
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    if args.cpu:
        cpu_check()
    else:
        if args.trace_dir is None and not args.diagnose_skewed_fc2:
            parser.error("the gfx942 qualification requires --trace-dir (new directory)")
        if int(os.getenv("WORLD_SIZE", "1")) != 1:
            parser.error("the native expert probe is single-rank")
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        # Import exactly the checkout containing this entrypoint, not a
        # Megatron checkout bundled in the image or on inherited PYTHONPATH.
        sys.path.insert(0, str(SOURCE_ROOT))
        torch.cuda.set_device(local_rank)
        torch.set_float32_matmul_precision("highest")
        device = torch.device("cuda", local_rank)
        if args.diagnose_skewed_fc2:
            diagnose_skewed_fc2(device)
        else:
            native_check(device, args.trace_dir)


if __name__ == "__main__":
    main()
