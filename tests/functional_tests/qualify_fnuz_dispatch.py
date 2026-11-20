"""Deterministic gfx942 codec and EP16/CXI dispatch qualification (no model or experts).

Run from the frozen Megatron snapshot with image 910 and the site's approved launcher.
Use one gfx942 rank for `codec`, and 16 block-placed ranks (4x4) for `ep16`.
All comparisons are exact; see the companion qualification contract.
"""

import argparse
from datetime import timedelta
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

EP = 16
EXPERTS = 256
LOCAL = EXPERTS // EP
TOKENS = 2 * 4096
TOPK = 8
TAG_DIVISOR = float(2**22)
UCCL = "52d4184c89deb5631be6dac6a9b1c40896c9e26e"
VERSIONS = {
    "torch": "2.14.0+rocm10.1.0",
    "triton": "3.8.0+git669b31ac.rocm10.1.0",
    "uccl": "0.1.1",
    "deep_ep": "0.1.0",
}
WRAPPER = "d30ad5231f476c726c246240eec9b6a2f9124e16752900ca562653623e77a768"
PATCHES = {
    "0001-alps-cxi-rocm-ep.patch": "d590f34fdf41a3c8582bb5237cde125485a4d6208c41d405bdf86e1d01a42d00",
    "0002-preserve-torch-resource-ownership.patch": "36e04b242db03a0fd24ce123483ebc3dd8d708de7ecce49b618a3f8f2072971b",
}


def equal(torch, actual, expected, label):
    """Fail on the first mismatch, retaining shape/dtype and mismatch count."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{label}: {actual.shape}/{actual.dtype} != {expected.shape}/{expected.dtype}")
    if not torch.equal(actual, expected):
        raise AssertionError(f"{label}: {int((actual != expected).sum())} unequal entries")


def levels(torch, device):
    """Positive E4M3FNUZ codes: bias 8, 3 mantissa bits, 2^-10 subnormals."""
    code = torch.arange(128, device=device, dtype=torch.int32)
    exp = code >> 3
    mantissa = (code & 7).float() / 8
    return torch.ldexp(mantissa + (exp != 0).float(), (exp - 8).clamp_min(-7))


def reference(torch, x):
    """FP32 scale, reciprocal and multiply; table-based RNE (no torch FP8 casts)."""
    block = x.float().reshape(x.shape[0], x.shape[1] // 128, 128)
    # Use a bool nonzero test on the BF16 storage, including BF16 subnormals.
    nonzero = ((x.contiguous().view(torch.int16) & 0x7FFF) != 0).reshape(block.shape).any(-1)
    amax = block.abs().amax(-1)
    sf = torch.where(nonzero, (amax.double() / 240.0).float().clamp_min(1e-30), 1.0)
    sf = torch.where(amax.isfinite(), sf, float("nan"))
    norm = (block * (1.0 / sf.double()).float()[..., None]).clamp(-240, 240)
    table = levels(torch, x.device)
    midpoints = (table[:-1] + table[1:]) * 0.5
    mag = norm.abs().contiguous()
    low = torch.searchsorted(midpoints, mag, out_int32=True).clamp_max(127)
    code = low + ((low < 127) & ((low & 1) == 1) & (mag == midpoints[low.clamp_max(126)]))
    negative = (code != 0) & (norm < 0)
    bits = (code + 128 * negative).to(torch.uint8)
    bits = torch.where(norm.isfinite(), bits, 128)
    decoded = (torch.where(negative, -table[code], table[code]) * sf[..., None]).to(torch.bfloat16)
    return bits.reshape_as(x), sf, decoded.reshape_as(x)


def activations(torch, key, width):
    """BF16 rows covering signs, zeros, powers-of-two scales and adjacent FP8 midpoints."""
    col = torch.arange(width, device=key.device, dtype=torch.int32)[None, :]
    key = key[:, None]
    group, lane = col // 128, col % 128
    ordered = (key * 47 + (col + 3) * 37 + group * 9) % 127
    table = levels(torch, key.device)
    value = torch.where(lane % 3 == 0, (table[ordered] + table[ordered + 1]) * 0.5, table[ordered])
    value = torch.where((key + col) % 2 == 0, value, -value)
    value = torch.where(lane == 0, 240.0, value)
    value = torch.where((key + group * 3) % 23 == 0, 0.0, value)
    return torch.ldexp(value, ((key * 7 + group * 3) % 15) - 8).to(torch.bfloat16)


def codec(torch, device):
    """Compare codec edge cases and no-fallback contracts on a single device."""
    from megatron.core.transformer.moe.fp8_jit import cast_dispatch_fnuz, dequant_dispatch_fnuz

    for width in (1792, 2048):
        rows = torch.arange(13, device=device, dtype=torch.int32)
        x = activations(torch, rows, width)
        x[0, :128] = -0.0
        x[1, :128] = 0.0
        x[1, :5] = torch.tensor([1e-32, -1e-32, 1e-28, 1e-35, 9.183549615799121e-41],
                                device=device, dtype=torch.bfloat16)
        x[2, :128] = 0.0
        x.view(torch.int16)[2, 0] = 1  # minimum nonzero BF16
        x[3, :128] = 0.0
        x[3, :3] = torch.tensor([100.0, -17.0, 0.001], device=device, dtype=torch.bfloat16)
        x[4, :128] = 0.0
        x[4, 0] = float("inf")
        x[5, :128] = 0.0
        x[5, 0] = float("nan")
        for src in (x, x.T.contiguous().T, x[:0]):
            expected, sf, decoded = reference(torch, src)
            payload, scales = cast_dispatch_fnuz(src)
            assert payload.dtype == torch.float8_e4m3fnuz and payload.is_contiguous()
            assert scales.stride() == (width // 128, 1)
            equal(torch, payload.view(torch.uint8), expected, "codec payload bits")
            finite = sf.isfinite()
            equal(torch, scales.isfinite(), finite, "codec nonfinite scales")
            equal(torch, scales.view(torch.int32)[finite], sf.view(torch.int32)[finite], "codec scale bits")
            actual = dequant_dispatch_fnuz(payload, scales)
            finite = decoded.isfinite()
            equal(torch, actual.isnan(), decoded.isnan(), "codec decode NaNs")
            equal(torch, actual.view(torch.int16)[finite], decoded.view(torch.int16)[finite], "codec BF16 bits")
        for invalid in (
            lambda: cast_dispatch_fnuz(x, use_packed_ue8m0=True),
            lambda: cast_dispatch_fnuz(x, use_ue8m0=True),
            lambda: cast_dispatch_fnuz(x.float()),
            lambda: cast_dispatch_fnuz(x[:, :-1]),
            lambda: cast_dispatch_fnuz(torch.empty((0, width), device="cpu", dtype=torch.bfloat16)),
            lambda: dequant_dispatch_fnuz(
                torch.empty_like(payload, dtype=torch.float8_e4m3fn), scales
            ),
            lambda: dequant_dispatch_fnuz(payload, scales.to(torch.int32)),
        ):
            try:
                invalid()
            except ValueError:
                continue
            raise AssertionError("invalid FNUZ request did not fail closed")
    return {"shapes": [[13, 1792], [13, 2048], [0, 1792], [0, 2048]], "exact": True}


def routes(torch, key, ragged, epoch):
    """Unique experts per row, repeated ranks; rank 14 sends nothing, rank 15 receives nothing in ragged."""
    source, token = key[:, None] // TOKENS, key[:, None] % TOKENS
    slot = torch.arange(TOPK, device=key.device)[None, :]
    if ragged:
        dest = (source * 3 + token * 7 + 5 * (slot // 3) + epoch) % 15
        dest = torch.where(token % 11 == 0, 0, dest)
        active = (source != 14) & (slot < (token % TOPK + 1))
        local_expert = slot
    else:
        dest = (source + token + 3 * (slot // 2) + epoch) % EP
        active = True
        local_expert = (token + slot) % LOCAL
    return torch.where(torch.as_tensor(active, device=key.device), dest * LOCAL + local_expert, -1)


def tags(torch, key, experts):
    slot = torch.arange(TOPK, device=key.device)[None, :]
    return torch.where(experts >= 0, ((key[:, None] + 1) * TOPK - slot).float() / TAG_DIVISOR, 0)


def token_signal(torch, keys, width, dest):
    col = torch.arange(width, device=keys.device, dtype=torch.int32)[None, :]
    return ((((keys[:, None] + col + dest) % 11) - 5) * 0.125).to(torch.bfloat16)


def probability_signal(torch, keys, dest):
    slot = torch.arange(TOPK, device=keys.device)[None, :]
    return dest * 32 + slot + 1.0 + (keys[:, None] % 19) / 32.0


def transport_identity():
    """Fail closed on image drift, including patched compatibility source."""
    import deep_ep
    import deep_ep.buffer
    import uccl.ep
    from megatron.core.transformer.moe import fused_a2a

    def digest(path):
        return sha256(Path(path).read_bytes()).hexdigest()

    root = Path("/opt/alps/apertus2-rocm")
    versions = {name: metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS or deep_ep.Buffer is not fused_a2a.Buffer:
        raise AssertionError(f"installed stack identity differs: {versions}")
    wrapper = Path(deep_ep.buffer.__file__).resolve()
    extension = Path(uccl.ep.__file__).resolve()
    site = Path("/opt/venv/lib/python3.12/site-packages")
    if (wrapper != site / "deep_ep/buffer.py" or extension != site / "uccl/ep.abi3.so"
            or digest(wrapper) != WRAPPER or uccl.ep.num_max_nvl_peers() != 4):
        raise AssertionError("installed UCCL Buffer/extension identity differs")
    if f"uccl https://github.com/uccl-project/uccl.git {UCCL}" not in (root / "source-pins.txt").read_text().splitlines():
        raise AssertionError("UCCL pin differs")
    if any(digest(root / "patches/uccl" / name) != checksum for name, checksum in PATCHES.items()):
        raise AssertionError("installed UCCL correction differs")
    return {"versions": versions, "uccl": UCCL, "wrapper_sha256": WRAPPER,
            "extension_sha256": digest(extension), "patches": PATCHES}


def ep16(torch, device, rank, record):
    """Exercise real scheduled quant/dispatch/dequant nodes and two outstanding handles."""
    from megatron.core.models.gpt.fine_grained_callables import (
        ManualGradNode, build_fp8_dispatch_node_callables, build_transformer_layer_callables,
    )
    from megatron.core.transformer.moe import fused_a2a
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher

    dist = torch.distributed
    hosts = [None] * EP
    dist.all_gather_object(hosts, socket.gethostname())
    assert len(set(hosts)) == 4 and all(hosts[j:j + 4] == [hosts[j]] * 4 for j in range(0, EP, 4))
    record["rank_hosts"] = hosts
    config = SimpleNamespace(
        moe_fp8_dispatch_fnuz=True, moe_use_fp8_dispatch=True, overlap_moe_expert_parallel_comm=True,
        moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="deepep", moe_router_topk=TOPK,
        num_moe_experts=EXPERTS, moe_router_dtype="fp32", moe_expert_capacity_factor=1.0,
        moe_permute_fusion=False, moe_deepep_num_sms=20, fp8=None, fp4=None, cuda_graph_scope=(),
    )
    group = dist.group.WORLD
    # Expert-TP is 1. new_group is collective even for groups with a single member.
    singles = [dist.new_group([i]) for i in range(EP)]
    dispatcher = MoEFlexTokenDispatcher(
        LOCAL, list(range(rank * LOCAL, (rank + 1) * LOCAL)), config,
        SimpleNamespace(ep=group, tp_ep=group, expt_tp=singles[rank]),
    )
    mlp = object.__new__(MoELayer)
    torch.nn.Module.__init__(mlp)
    mlp.token_dispatcher = dispatcher
    mlp._preload_expert_weights = lambda: None
    layer = SimpleNamespace(config=config, mlp=mlp, _forward_mlp=lambda: None,
                            init_backward_dw_wrapper=lambda: None, backward_dw_wrapper=None)
    functions = build_fp8_dispatch_node_callables(layer)
    dispatch = build_transformer_layer_callables(layer)[0][1]
    flags = dict(config=config, is_moe=True, num_local_experts=LOCAL)
    compute, comm = torch.cuda.Stream(), torch.cuda.Stream()
    manager = dispatcher._comm_manager
    keys = torch.arange(rank * TOKENS, (rank + 1) * TOKENS, dtype=torch.int32, device=device)
    all_keys = torch.arange(EP * TOKENS, dtype=torch.int32, device=device)
    fields = ("num_sms", "num_max_nvl_chunked_send_tokens", "num_max_nvl_chunked_recv_tokens",
              "num_max_rdma_chunked_send_tokens", "num_max_rdma_chunked_recv_tokens")

    def configs():
        getters = (fused_a2a.Buffer.get_dispatch_config, fused_a2a.Buffer.get_combine_config)
        result = tuple(tuple(int(getattr(getter(EP), k)) for k in fields) for getter in getters)
        assert result == ((20, 36, 288, 20, 140), (20, 4, 288, 12, 132)), result
        return result

    record["config_fields"], record["configs"] = fields, configs()
    record["cases"] = []

    def forward(width, ragged, epoch):
        print(f"FNUZ_TRANSPORT_CASE forward width={width} epoch={epoch} rank={rank}", flush=True)
        event = torch.cuda.Event()
        state = SimpleNamespace()

        def node(name, stream, fw, bw):
            return ManualGradNode(stream, event, state, None, fw, name=name,
                                  backward_submodule=bw, extra_args=flags)

        quant = node("moe_pre_dispatch", compute, *functions["moe_pre_dispatch"])
        exchange = node("moe_dispatch", comm, dispatch, functions["moe_dispatch_backward"])
        decode = node("moe_post_dispatch", compute, *functions["moe_post_dispatch"])
        outgoing = routes(torch, keys, ragged, epoch)
        probs = tags(torch, keys, outgoing)
        # Valid columns are unique. Use an out-of-range scratch column for dropped
        # slots so those zero writes cannot clobber expert 0.
        scores = torch.zeros((TOKENS, EXPERTS + 1), device=device)
        scores.scatter_(1, torch.where(outgoing < 0, EXPERTS, outgoing), probs)
        scores = scores[:, :EXPERTS].contiguous().requires_grad_()
        source_tokens = activations(torch, keys, width).requires_grad_()
        source_tokens, local_probs = dispatcher.dispatch_preprocess(source_tokens, scores > 0, scores)
        equal(torch, manager.token_indices, outgoing, "source order / dropped indices")
        event.record()
        tx = quant.forward((source_tokens, None, local_probs))
        rx = exchange.forward(tx)
        got, none = decode.forward(rx)
        assert none is None
        event.wait()

        # Check the local encoder independently, then check wire bit preservation
        # by association against regenerated source rows, before trusting BF16 decode.
        expected, sf, _ = reference(torch, source_tokens.detach())
        equal(torch, tx[0].detach().view(torch.uint8), expected, "EP source FP8 bits")
        equal(torch, tx[1].view(torch.int32), sf.view(torch.int32), "EP source scale bits")
        route_all = routes(torch, all_keys, ragged, epoch)
        expected_incoming = ((route_all >= rank * LOCAL) & (route_all < (rank + 1) * LOCAL))
        expected_keys = all_keys[expected_incoming.any(-1)]
        recv_p = state.dispatched_probs
        recv_i = manager.dispatched_indices
        # Tags in slots 1..7 are below their (key+1)*TOPK boundary.
        recv_key = ((recv_p.detach().amax(-1) * TAG_DIVISOR).to(torch.int32) - 1) // TOPK
        equal(torch, recv_key.sort().values, expected_keys, "one receive per expected source/token")
        exp_route = route_all[recv_key.long()]
        in_slot = expected_incoming[recv_key.long()]
        equal(torch, recv_i, torch.where(in_slot, exp_route - rank * LOCAL, -1), "receiver expert slots")
        exp_probs = tags(torch, recv_key, exp_route)
        equal(torch, recv_p.detach(), exp_probs.masked_fill(~in_slot, 0), "receiver probability slots")
        counts = (route_all[..., None] == (rank * LOCAL + torch.arange(LOCAL, device=device))).sum((0, 1))
        equal(torch, manager.tokens_per_expert.long(), counts.cpu(), "received expert counts")
        assert rx[0].dtype == torch.float8_e4m3fnuz and rx[1].stride() == (width // 128, 1)
        assert got.dtype == torch.bfloat16
        for start in range(0, recv_key.numel(), 1024):
            sl = slice(start, start + 1024)
            bits, sf, bf16 = reference(torch, activations(torch, recv_key[sl], width))
            equal(torch, rx[0].detach()[sl].view(torch.uint8), bits, "received FP8 bits")
            equal(torch, rx[1][sl].view(torch.int32), sf.view(torch.int32), "received scale bits")
            equal(torch, got.detach()[sl].view(torch.int16), bf16.view(torch.int16), "received BF16 bits")

        # Synthetic expert boundary: real leaf gradient, including a (0,8) leaf
        # on the zero-receive rank. Use a detached BF16 signal like the schedule.
        grad_tokens = token_signal(torch, recv_key, width, rank)
        grad_probs = probability_signal(torch, recv_key, rank).masked_fill(~in_slot, 0)
        (recv_p * grad_probs).sum().backward()
        equal(torch, recv_p.grad, grad_probs, "detached probability leaf gradient")
        combine_input = grad_tokens.detach().requires_grad_()
        comm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(comm):
            combined = dispatcher.token_combine(combine_input)
        torch.cuda.current_stream().wait_stream(comm)
        assert manager.handle is None  # forward combine releases the module slot
        assert state.dispatch_handle is not None  # backward still owns this microbatch's handle
        exp_tokens = torch.zeros_like(source_tokens, dtype=torch.float32)
        dest = outgoing // LOCAL
        for dst in range(EP):
            exp_tokens.add_(token_signal(torch, keys, width, dst) * (dest == dst).any(-1)[:, None])
        exp_tokens = exp_tokens.to(torch.bfloat16)
        exp_grad_probs = probability_signal(torch, keys, dest).masked_fill(outgoing < 0, 0)
        equal(torch, combined, exp_tokens, "BF16 forward combine")
        send_counts = [(int((dest == dst).any(-1).sum())) for dst in range(EP)]
        recv_counts = torch.bincount(recv_key.long() // TOKENS, minlength=EP).tolist()
        if ragged:
            assert (sum(send_counts) == 0) == (rank == 14)
            assert (sum(recv_counts) == 0) == (rank == 15)
        else:
            assert all(send_counts) and all(recv_counts)
        summary = dict(width=width, epoch=epoch, ragged=ragged, send_counts=send_counts,
                       recv_counts=recv_counts, topk=TOPK, scale_row_bytes=width // 128 * 4,
                       empty_experts=int((counts == 0).sum()))
        return SimpleNamespace(quant=quant, exchange=exchange, decode=decode, state=state, recv_key=recv_key,
                               grad_tokens=grad_tokens, exp_tokens=exp_tokens, exp_grad_probs=exp_grad_probs,
                               combined=combined, combine_input=combine_input, summary=summary)

    def backward(case):
        width, epoch = case.summary["width"], case.summary["epoch"]
        print(f"FNUZ_TRANSPORT_CASE backward width={width} epoch={epoch} rank={rank}", flush=True)
        (combine_grad,) = torch.autograd.grad(case.combined, case.combine_input,
                                              token_signal(torch, keys, width, EP))
        torch.cuda.current_stream().wait_stream(comm)
        equal(torch, combine_grad, token_signal(torch, case.recv_key, width, EP), "cached BF16 dispatch")
        case.decode.event.record()
        grad_in = case.quant.backward(case.exchange.backward(case.decode.backward((case.grad_tokens, None))))
        case.quant.event.wait()
        assert grad_in[0].dtype == torch.bfloat16 and grad_in[1] is None
        equal(torch, grad_in[0], case.exp_tokens, "BF16 reverse combine")
        equal(torch, grad_in[2], case.exp_grad_probs, "reverse probability gradients")
        record["cases"].append(dict(case.summary, exact=True))

    a = forward(1792, False, 0)
    b = forward(1792, True, 1)
    assert a.state.dispatch_handle is not b.state.dispatch_handle
    buffer = fused_a2a._buffer
    assert buffer.runtime.get_num_rdma_ranks() == 4
    hints = [get(EP) for get in (buffer.get_dispatch_config, buffer.get_combine_config)]
    assert buffer.num_nvl_bytes == max(c.get_nvl_buffer_size_hint(1792 * 2, EP) for c in hints)
    assert buffer.num_rdma_bytes == max(c.get_rdma_buffer_size_hint(1792 * 2, EP) for c in hints)
    record["width1792_buffer"] = dict(nvl=buffer.num_nvl_bytes, rdma=buffer.num_rdma_bytes)
    backward(a)
    backward(b)
    assert fused_a2a._buffer is buffer
    del a, b
    # 12 FP32 scales = 48 bytes, the aligned branch; keep using the original
    # width-1792 buffer rather than introducing communicator reallocation.
    backward(forward(1536, False, 2))
    assert fused_a2a._buffer is buffer
    record["final_configs"] = configs()
    torch.cuda.synchronize()
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("codec", "ep16"))
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    node = int(os.environ.get("SLURM_NODEID", str(rank // 4)))
    record = dict(status="fail", rank=rank, local=local, node=node, world=world, phase=args.phase)
    print(f"FNUZ_TRANSPORT_START {json.dumps(record)}", flush=True)
    try:
        import torch

        if not __debug__:
            raise RuntimeError("Run FNUZ qualification without Python optimization")
        assert (args.phase == "codec" and world == 1 and rank == 0) or (
            args.phase == "ep16" and world == EP and rank == 4 * node + local
            and 0 <= local < 4 and 0 <= node < 4
        )
        assert torch.version.hip and 0 <= local < torch.cuda.device_count()
        if args.phase == "ep16":
            assert torch.cuda.device_count() == 4
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        record["device"] = torch.cuda.get_device_properties(device).gcnArchName
        assert record["device"].split(":")[0] == "gfx942"
        record["stack"] = transport_identity()
        if args.phase == "codec":
            record["codec"] = codec(torch, device)
        else:
            fixed = {"UCCL_EP_TRANSPORT": "cxi", "UCCL_SOCKET_IFNAME": "hsn",
                     "UCCL_CXI_THREADING": "safe", "NCCL_P2P_DISABLE": "0"}
            assert all(os.environ.get(k) == v for k, v in fixed.items()), fixed
            assert not any(k in os.environ for k in ("UCCL_EP_DISPATCH_CONFIG", "UCCL_EP_COMBINE_CONFIG", "EFA_HOME"))
            record["environment"] = {k: v for k, v in os.environ.items() if k.startswith(("UCCL_", "FI_", "NCCL_"))}
            torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=3), device_id=device)
            ep16(torch, device, rank, record)
            torch.distributed.destroy_process_group()
        record["status"] = "pass"
    except BaseException as error:
        record["error"] = repr(error)
        raise
    finally:
        print(f"FNUZ_TRANSPORT_RESULT {json.dumps(record, sort_keys=True, allow_nan=False)}", flush=True)


if __name__ == "__main__":
    main()
