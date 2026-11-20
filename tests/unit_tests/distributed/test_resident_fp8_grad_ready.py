"""CPU contract checks for resident expert grad readiness and model weight invalidation."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed import distributed_data_parallel as ddp
from megatron.core.optimizer import optimizer as opt


@pytest.fixture
def ready(monkeypatch):
    param = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    param.main_grad = torch.zeros(2, dtype=torch.float32)
    timeline = []
    writer = object()
    final = SimpleNamespace(wait_stream=lambda stream: timeline.append(("wait", stream)))
    monkeypatch.setattr(ddp, "is_graph_capturing", lambda: False)
    monkeypatch.setattr(ddp, "get_num_microbatches", lambda: 2)
    monkeypatch.setattr(
        ddp.torch.cuda, "current_stream", lambda device=None: writer if device is not None else final
    )
    bucket = SimpleNamespace(
        is_last_microbatch=False,
        register_grad_ready=lambda p, force: timeline.append(("ready", p)),
        start_grad_sync=lambda: timeline.append(("reduce", param.main_grad.tolist())),
    )
    wrapper = SimpleNamespace(
        param_to_bucket_group={param: bucket},
        force_all_reduce=False,
        _resident_fp8_delayed_params=[param],
        _resident_fp8_grad_counts={},
        _resident_fp8_grad_streams=set(),
        bucket_groups=[],
        expert_parallel_bucket_groups=[bucket],
    )
    wrapper._wait_resident_fp8_grads = lambda: ddp.DistributedDataParallel._wait_resident_fp8_grads(
        wrapper
    )
    hook = ddp.DistributedDataParallel._make_resident_fp8_ready_hook(wrapper, param)
    return wrapper, param, hook, bucket, writer, timeline


def test_aligned_sync_waits_for_two_contributions_including_empty(ready):
    wrapper, param, hook, _, writer, timeline = ready
    param.main_grad.add_(torch.tensor([3.25, 0.125]))
    hook(param)
    with pytest.raises(RuntimeError, match="count 1, expected 2"):
        ddp.DistributedDataParallel.start_grad_sync(wrapper)
    assert timeline == []
    hook(param)  # Last microbatch is empty: complete without writing a gradient.
    ddp.DistributedDataParallel.start_grad_sync(wrapper)
    assert timeline == [("wait", writer), ("reduce", [3.25, 0.125])]
    assert param.grad is None
    with pytest.raises(RuntimeError, match="Duplicate"):
        hook(param)


def test_ready_rejects_wrong_param_and_unexpected_autograd_gradient(ready):
    _, param, hook, _, _, _ = ready
    with pytest.raises(RuntimeError, match="Invalid"):
        hook(torch.nn.Parameter(torch.zeros(2)))
    param.grad = torch.zeros_like(param)
    with pytest.raises(RuntimeError, match="Invalid"):
        hook(param)


def test_finalize_rejects_missing_contribution(ready):
    wrapper, param, hook, bucket, _, timeline = ready
    bucket.finish_grad_sync = lambda **kwargs: timeline.append(("finish", kwargs))
    hook(param)
    with pytest.raises(RuntimeError, match="count 1, expected 2"):
        ddp.DistributedDataParallel.finish_grad_sync(wrapper)
    assert timeline == []


def test_optimizer_model_data_copy_invalidates_the_resident_version(monkeypatch):
    w = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    original_version = w._version
    authoritative = torch.full((4,), 3.25)

    def copy(*, this, that, overflow_buf):
        that[0].copy_(this[0])
        assert w._version == original_version

    monkeypatch.setattr(opt, "_multi_tensor_copy_this_to_that", copy)
    wrapper = SimpleNamespace(
        _get_model_and_main_params_data_float16=lambda: ([w.data], [authoritative]),
        _dummy_overflow_buf=None,
        _resident_fp8_model_params=[w],
    )
    opt.Float16OptimizerWithFloat16Params._copy_main_params_to_model_params(wrapper)
    assert w._version == original_version + 1
    torch.testing.assert_close(w, authoritative.bfloat16())
