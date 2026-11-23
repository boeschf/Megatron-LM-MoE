"""Compare chunked LM head against the pinned Megatron native fused loss on TP1."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from torch import distributed as dist

from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.models.gpt.chunked_lm_head import chunked_lm_head_loss
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig


LIMITS = {
    "loss_abs": 5.0e-4,
    "dx": (0.01, 0.03),
    "dw": (0.015, 0.03),
}


def errors(actual, expected):
    """Calculate full-range metrics without holding full FP32 wgrad copies."""
    square, norm_sq, delta_max, expected_max = 0.0, 0.0, 0.0, 0.0
    for a, b in zip(actual.reshape(-1).split(2**20), expected.reshape(-1).split(2**20), strict=True):
        a, b = a.float(), b.float()
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        diff = a - b
        square += float(diff.square().sum())
        norm_sq += float(b.square().sum())
        delta_max = max(delta_max, float(diff.abs().max()))
        expected_max = max(expected_max, float(b.abs().max()))
    return (square / max(norm_sq, 1.0e-30)) ** 0.5, delta_max / max(expected_max, 1.0e-30)


def compare(device, shape, cap, *, measure_peaks=False):
    """Check output, dgrad, wgrad, forward saves and allocation peak."""
    seq, batch, hidden_size, vocab = shape
    generator = torch.Generator(device=device).manual_seed(1729)
    hidden = (torch.randn(seq, batch, hidden_size, generator=generator, device=device) * 0.25).bfloat16()
    weight = (torch.randn(vocab, hidden_size, generator=generator, device=device) * 0.0167).bfloat16()
    labels = torch.randint(vocab, (batch, seq), generator=generator, device=device)
    labels[0, 0], labels[-1, -1] = 0, vocab - 1
    upstream = torch.randn(batch, seq, generator=generator, device=device) / (batch * seq)
    upstream[:, ::7] = 0

    def run(chunked):
        x, w = hidden.detach().requires_grad_(), weight.detach().requires_grad_()
        logits_shapes, saved = [], []
        gradients_seen = []
        w.register_post_accumulate_grad_hook(
            lambda param: gradients_seen.append((param.grad.shape, param.grad.dtype))
        )

        def loss_fn(target, logits):
            logits_shapes.append((logits.shape[0] * logits.shape[1], logits.shape[2], logits.dtype))
            return fused_vocab_parallel_cross_entropy(
                logits, target.t().contiguous(), dist.group.WORLD
            ).t().contiguous()

        def record(tensor):
            saved.append(tensor.shape)
            return tensor

        if measure_peaks:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)
        with torch.autograd.graph.saved_tensors_hooks(record, lambda t: t):
            loss = chunked_lm_head_loss(x, w, labels, loss_fn, cap) if chunked else loss_fn(labels, x @ w.t())
        loss.backward(upstream)
        dx, dw = x.grad, w.grad
        assert gradients_seen == [(w.shape, w.dtype)], gradients_seen
        peak = None
        if measure_peaks:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device) - before
        if chunked:
            assert saved == [x.shape, w.shape, labels.shape], saved
            big_chunk = (cap // batch) * batch
            assert len(logits_shapes) == 2 * ((seq * batch + big_chunk - 1) // big_chunk)
            assert all(n <= cap and v == vocab and dt == torch.bfloat16 for n, v, dt in logits_shapes)
        assert dx.dtype == dw.dtype == torch.bfloat16
        return loss.detach(), dx, dw, peak

    # Match dense and chunked paths against identical inputs and masked upstream loss.
    base, bdx, bdw, dense_peak = run(False)
    result, dx, dw, chunk_peak = run(True)
    report = dict(shape=shape, chunk=cap, loss_abs=float((result - base).abs().max()),
                  dx=errors(dx, bdx), dw=errors(dw, bdw), dense_peak=dense_peak, chunk_peak=chunk_peak)
    print(json.dumps(report), flush=True)
    assert torch.isfinite(result).all() and report["loss_abs"] <= LIMITS["loss_abs"], report
    for key in ("dx", "dw"):
        assert all(seen <= limit for seen, limit in zip(report[key], LIMITS[key], strict=True)), report
    if measure_peaks:
        assert dense_peak - chunk_peak >= 3 * 2**30, report


def validate_config():
    """Ensure the default is unchanged and unsupported configurations fail closed."""
    basic = dict(num_layers=1, hidden_size=128, num_attention_heads=4, bf16=True,
                 params_dtype=torch.bfloat16, cross_entropy_loss_fusion=True,
                 gradient_accumulation_fusion=False)
    assert TransformerConfig(**basic).lm_head_token_chunk_size == 0
    assert TransformerConfig(**basic, lm_head_token_chunk_size=512).lm_head_token_chunk_size == 512
    for other in ({"lm_head_token_chunk_size": 513}, {"lm_head_token_chunk_size": -1},
                  {"cross_entropy_fusion_impl": "te"}, {"gradient_accumulation_fusion": True},
                  {"bf16": False}, {"use_mup": True}, {"tensor_model_parallel_size": 2}):
        try:
            TransformerConfig(**{**basic, "lm_head_token_chunk_size": 512, **other})
        except ValueError as exc:
            assert "Chunked LM head" in str(exc), exc
        else:
            raise AssertionError(f"unsupported configuration passed: {other}")


def check_gpt_route():
    """Test the opt-in training branch and leave inference/no-label output dense."""
    class Output(torch.nn.Linear):
        calls = 0
        sequence_parallel = False

        def forward(self, hidden, weight=None, runtime_gather_output=None):
            assert weight is None
            self.calls += 1
            return super().forward(hidden), None

    cfg = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4, bf16=True,
                            params_dtype=torch.bfloat16, cross_entropy_loss_fusion=True,
                            gradient_accumulation_fusion=False, lm_head_token_chunk_size=4)
    model = GPTModel.__new__(GPTModel)
    torch.nn.Module.__init__(model)
    model.config = cfg
    model.pg_collection = SimpleNamespace(tp=dist.group.WORLD)
    model.share_embeddings_and_output_weights = False
    model.pre_process = False
    model.post_process = True
    model.output_layer = Output(128, 257, bias=False, dtype=torch.bfloat16)
    hidden = torch.full((7, 2, 128), 0.125, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.arange(14).view(2, 7)

    def post(labels):
        return model._postprocess(hidden, None, None, labels, None, None, None)

    chunked_loss = post(labels)
    assert model.output_layer.calls == 0
    cfg.lm_head_token_chunk_size = 0
    dense_loss = post(labels)
    assert model.output_layer.calls == 1
    torch.testing.assert_close(chunked_loss, dense_loss, atol=LIMITS["loss_abs"], rtol=0)
    cfg.lm_head_token_chunk_size = 4
    assert post(None).shape == (2, 7, 257) and model.output_layer.calls == 2
    model.eval()
    torch.testing.assert_close(post(labels), dense_loss, atol=LIMITS["loss_abs"], rtol=0)
    assert model.output_layer.calls == 3
    model.train()
    model.share_embeddings_and_output_weights = True
    try:
        post(labels)
    except ValueError as exc:
        assert "untied" in str(exc)
    else:
        raise AssertionError("tied head was not rejected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    assert os.environ.get("WORLD_SIZE", "1") == "1"
    validate_config()
    if args.device == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        assert torch.version.hip and torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] == "gfx942"
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", rank=0, world_size=1)
        try:
            compare(device, (4096, 2, 3584, 200064), 512, measure_peaks=True)
        finally:
            dist.destroy_process_group()
    else:
        with TemporaryDirectory() as tmp:
            dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{Path(tmp) / 'rdzv'}")
            try:
                check_gpt_route()
                for shape, cap in (((17, 2, 128, 257), 8), ((29, 3, 128, 8201), 15),
                                   ((8, 2, 128, 200064), 8), ((33, 1, 64, 1025), 512)):
                    compare(torch.device("cpu"), shape, cap)
            finally:
                dist.destroy_process_group()
    print("Chunked LM head native loss, gradients and bounded saves: PASS", flush=True)


if __name__ == "__main__":
    main()
