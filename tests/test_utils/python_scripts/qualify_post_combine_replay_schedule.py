"""Image-910 local schedule A/B; no router, expert kernels, or EP traffic."""

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.extensions.transformer_engine import TELinear, TENorm
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.common.model_chunk_schedule_plan import TransformerLayerSchedulePlan
from megatron.core.pipeline_parallel.utils import get_comm_stream, set_streams
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher
from megatron.core.transformer.transformer_layer import TransformerLayer


LATENT, HIDDEN = (4096, 2, 1792), (4096, 2, 3584)
IN_FLIGHT = 4


def arm(replay, fixture):
    """Fill four real layer plans, interleave one more forward, and drain."""
    dtype = torch.bfloat16
    torch.manual_seed(42)
    tensor_parallel.model_parallel_cuda_manual_seed(42)
    # Only the local settings, without admitting a synthetic topology to training.
    local = ModelParallelConfig(bf16=True, params_dtype=dtype, delay_wgrad_compute=True)
    config = SimpleNamespace(
        **vars(local), moe_latent_size=LATENT[-1], normalization="RMSNorm",
        layernorm_zero_centered_gamma=False, fused_residual_rmsnorm=False,
        disable_parameter_transpose_cache=False, init_model_with_meta_device=False,
        moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="deepep",
        moe_use_fp8_dispatch=False, cuda_graph_scope=[], fp8=None, fp4=None,
        bias_dropout_fusion=True, moe_recompute_post_combine=replay,
    )
    moe = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(moe)
    moe.config = config
    moe.num_local_experts = 2
    moe.fc2_latent_proj = projection = TELinear(
        LATENT[-1], HIDDEN[-1], parallel_mode="duplicated", config=config,
        init_method=lambda w: torch.nn.init.normal_(w, std=0.0167038), bias=False,
        skip_bias_add=False, skip_weight_param_allocation=False, is_expert=False,
    )
    queue = projection.wgrad_store.context
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config, layer.mlp = config, moe
    layer.post_mlp_layernorm = norm = TENorm(config, HIDDEN[-1], eps=1e-5)
    layer.residual_output_scale = (2 * 42) ** -0.5
    layer.keel, layer.offload_mlp_norm, layer.hidden_dropout = False, False, 0.0
    layer.bias_dropout_add_exec_handler, layer.mlp_bda = nullcontext, get_bias_dropout_add
    layer.init_backward_dw_wrapper = lambda: None
    layer.backward_dw_wrapper = SimpleNamespace(backward_dw=lambda: None)
    comp, comm = torch.cuda.current_stream(), get_comm_stream()
    assert comp != comm
    chunk = SimpleNamespace(model=SimpleNamespace(decoder=SimpleNamespace(final_layernorm=None)))
    events, sums = [], []

    def record(name):
        assert torch.cuda.current_stream() == comm
        events.append(name)

    dispatcher = SimpleNamespace(hidden_shape=None)

    def reshape(x):
        record("reshape")
        shape, dispatcher.hidden_shape = dispatcher.hidden_shape, None
        assert shape == LATENT  # Cannot be used after the original forward.
        return MoEFlexTokenDispatcher.combine_postprocess(SimpleNamespace(hidden_shape=shape), x)

    def project(x):
        record("project_grad" if torch.is_grad_enabled() else "project_no_grad")
        return projection_forward(x)

    def drain():
        record("drain")
        assert queue.qsize() == 1 and projection.weight.grad is None
        projection_dw()
        assert queue.empty() and projection.weight.grad is not None

    projection_forward, projection_dw = projection.forward, projection.backward_dw
    projection.forward, projection.backward_dw = project, drain
    dispatcher.combine_postprocess = reshape
    moe.token_dispatcher = dispatcher
    expert_drains = []
    moe.experts = SimpleNamespace(backward_dw=lambda: expert_drains.append(1))

    def on_norm_forward(_, inputs, y):
        summed, size = inputs[0], y.numel() * y.element_size()
        assert summed.dtype == dtype and y.dtype == dtype
        assert summed.untyped_storage().nbytes() == size
        record("norm_forward")
        sums.append(summed.untyped_storage())

        def norm_backward(grad):
            record("norm_backward")
            assert summed.untyped_storage().nbytes() == size

        y.register_hook(norm_backward)

    norm.register_forward_hook(on_norm_forward)

    def attention(node, x, shared, residual):
        node.layer_state.shared_expert_output = node.detach(shared)
        node.layer_state.residual = node.detach(residual)
        return x.clone()

    def combine(node, x):
        record("combine_forward")
        dispatcher.hidden_shape = LATENT
        return x.flatten(0, 1).clone()

    def plan():
        event = torch.cuda.Event()
        event.record(comp)
        p = TransformerLayerSchedulePlan(layer, event, chunk, comp, comm, extra_args={})
        assert p.moe_combine.stream == p.moe_post_combine.stream == comm
        p.attn.submodule = attention
        p.moe_dispatch.submodule = lambda node, x: x.clone()
        p.mlp.submodule = lambda node, x: x.clone()
        p.moe_combine.submodule = combine
        backward = p.moe_combine.backward_impl

        def reverse(outputs, grads):
            record("combine_backward")
            assert queue.qsize() == 1 and projection.weight.grad is None
            return backward(outputs, grads)

        p.moe_combine.backward_func = reverse
        return p

    plans = [plan() for _ in fixture]
    run = TransformerLayerSchedulePlan.run
    forward = ["combine_forward", "reshape", "project_no_grad" if replay else "project_grad",
               "norm_forward"]
    backward = (["project_grad"] if replay else []) + ["norm_backward", "combine_backward", "drain"]
    saved_sum_bytes = 0 if replay else 2 * LATENT[0] * LATENT[1] * HIDDEN[2]
    values = []
    for p, data in zip(plans[:IN_FLIGHT], fixture[:IN_FLIGHT], strict=True):
        y, _ = run(p, None, f_input=data[:3])
        values.append(y)
    assert queue.empty() and events == forward * IN_FLIGHT
    assert [storage.nbytes() for storage in sums] == [saved_sum_bytes] * IN_FLIGHT
    sums.clear()
    events.clear()

    receipts = []
    for i, (b, data) in enumerate(zip(plans, fixture, strict=True)):
        f = plans[-1] if i == 0 else None
        y, grads = run(f, b, f_input=fixture[-1][:3] if f else None, b_grad=data[-1])
        if f is not None:
            values.append(y)
        assert [storage.nbytes() for storage in sums] == ([saved_sum_bytes] if f else [])
        sums.clear()
        assert events == backward + (forward if f is not None else [])
        assert len(expert_drains) == i + 1 and queue.empty()
        events.clear()
        # The fixture, unlike the training path, syncs before reading host receipts.
        torch.cuda.synchronize()
        entries = (values[i], *grads, projection.weight.grad, norm.weight.grad)
        assert len(entries) == 6 and all(t is not None and torch.isfinite(t).all().item() for t in entries)
        receipts.append(tuple(t.detach().cpu().contiguous().view(torch.uint8) for t in entries))
        projection.weight.grad = norm.weight.grad = None
    return receipts


def main():
    """Require bitwise local schedule parity using the actual TE delayed projection."""
    if (
        (torch.__version__, torch.version.hip, torch.cuda.device_count())
        != ("2.14.0+rocm10.1.0", "7.16.26385", 1)
        or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx942"
    ):
        raise ValueError("expected one image-910 MI300A")
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=120))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        set_streams()
        rng = torch.Generator(device="cuda").manual_seed(1701)
        fixture = [
            tuple(torch.randn(s, device="cuda", dtype=torch.bfloat16, generator=rng)
                  .requires_grad_(j < 3)
                  for j, s in enumerate((LATENT, HIDDEN, HIDDEN, HIDDEN)))
            for _ in range(IN_FLIGHT + 1)
        ]
        baseline, replay = arm(False, fixture), arm(True, fixture)
        for i, (a, b) in enumerate(zip(baseline, replay, strict=True)):
            for name, x, y in zip(
                ("output", "latent", "shared", "residual", "projection_wgrad", "norm_wgrad"),
                a, b, strict=True,
            ):
                if not torch.equal(x, y):
                    raise ValueError(f"bitwise mismatch for {name} on microbatch {i}")
        print("Five local schedule microbatches passed bitwise parity and exact delayed drain.")
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
