"""Compare the native optimizer on mapped expert parameters, without FP8."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import md_decoupling as md


@pytest.fixture
def tp1(tmp_path):
    if torch.distributed.is_initialized():
        raise RuntimeError("Run the expert optimizer mapping check in a standalone process.")
    torch.distributed.init_process_group(
        "gloo", init_method=(tmp_path / "rdzv").as_uri(), rank=0, world_size=1
    )
    try:
        yield SimpleNamespace(tp=torch.distributed.group.WORLD, expt_tp=torch.distributed.group.WORLD)
    finally:
        torch.distributed.destroy_process_group()


def check_mapping(pg, *, fc1, rows, cols, experts=2, device="cpu"):
    """Compare identical supplied FP32 main gradients over three optimizer updates."""
    generator = torch.Generator(device="cpu").manual_seed(16929 + int(fc1))
    # Match authoritative BF16 initialization before creating FP32 main parameters.
    initial = (torch.randn((experts, rows, cols), generator=generator) * 0.0167038)
    initial = initial.to(device=device, dtype=torch.bfloat16).float()
    fused = torch.nn.Parameter(initial.clone())
    fused.merged_offload_expert = True
    resident = [torch.nn.Parameter(w.clone()) for w in initial.unbind(0)]
    for param in resident:
        # TEGroupedLinear explicitly stamps partition_dim; at ETP1 the group has one rank.
        param.partition_dim = 0 if fc1 else 1
    for index, p in enumerate([fused, *resident]):
        layer = "linear_fc1" if fc1 else "linear_fc2"
        name = (
            f"decoder.layers.0.mlp.experts.weight{1 if fc1 else 2}"
            if p is fused
            else f"decoder.layers.0.mlp.experts.{layer}.weight{index - 1}"
        )
        p.expert_tp = True
        p.is_out_proj = not fc1
        p.glu_split_dim = md._glu_fc1_split_dim(name, p, True)
        assert p.glu_split_dim == ((1 if p is fused else 0) if fc1 else None)

    def optimizer(params):
        return md.MDDecoupling(
            [dict(params=params, min_lr=0.0001, max_lr=0.0308)],
            lr=0.0308,
            weight_decay=0,
            betas=(0.9, 0.95),
            use_orthogonal_updates=True,
            momentum_beta=0.95,
            use_nesterov=True,
            coefficient_type="polar_express",
            num_ns_steps=5,
            scale_mode="shape_up",
            tp_mode="duplicated",
            pg_collection=pg,
            hypersphere_mode="row",
            hypersphere_radius_mode="fan_in",
            hypersphere_gains_mode="rowcol",
            gain_parametrization="softplus",
            gains_lr=0.00616,
            gains_min_lr=0.0001,
            gains_betas=(0.9, 0.95),
        )

    reference = optimizer([fused])
    candidate = optimizer(resident)
    layout_ref = reference._gain_layout(fused)
    layout_te = candidate._gain_layout(resident[0])
    for axis in ("row", "col"):
        assert layout_ref[f"{axis}_shape"] == (experts, *layout_te[f"{axis}_shape"])
        assert layout_ref[f"{axis}_sum"] == layout_te[f"{axis}_sum"] + 1
        assert layout_ref[f"{axis}_norm"] == layout_te[f"{axis}_norm"] + 1
    logical_fused, _ = reference._logical_blocks(fused, fused, False, True)
    logical_resident = [
        matrix
        for p in resident
        for matrix in candidate._logical_blocks(p, p, False, False)[0]
    ]
    assert len(logical_fused) == len(logical_resident) == experts * (2 if fc1 else 1)
    for a, b in zip(logical_fused, logical_resident):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=2e-6)
        torch.testing.assert_close(a.norm(dim=1), torch.ones_like(a[:, 0]), atol=1e-6, rtol=0)

    # Exercise differing LR and the carry-over of matrix momentum and every gain moment.
    # One expert has a true zero contribution on the second step, not a missing gradient.
    tolerances = dict(atol=3e-7, rtol=2e-5)
    for step, lr in enumerate((0.000308, 0.000616, 0.000924), 1):
        grad = torch.randn(fused.shape, generator=generator).to(device) * 0.012
        grad[0, :, ::3] *= -2
        if step == 2:
            grad[-1].zero_()
        fused.grad = grad.clone()
        for p, g in zip(resident, grad.unbind(0)):
            p.grad = g.clone()
        for opt in (reference, candidate):
            opt.param_groups[0]["lr"] = lr
            opt.step()
            assert opt.param_groups[0]["step"] == step
        for index, p in enumerate(resident):
            torch.testing.assert_close(fused[index], p, **tolerances)
            # Also compare the actual effective BF16 values seen by the model, allowing
            # one BF16 rounding step around a bin boundary after the strict FP32 check.
            torch.testing.assert_close(
                fused[index].bfloat16(), p.bfloat16(), atol=2e-6, rtol=0.008
            )
            got, want = candidate.state[p], reference.state[fused]
            assert got.keys() == want.keys() == {
                "exp_avg", "row_gain", "row_gain_m", "row_gain_v",
                "col_gain", "col_gain_m", "col_gain_v",
            }
            for name in want:
                torch.testing.assert_close(want[name][index], got[name], **tolerances)
            for param, state, optimizer_ in ((p, got, candidate), (fused[index], want, reference)):
                r, c = state["row_gain"], state["col_gain"]
                if param is not p:
                    r, c = r[index], c[index]
                bare = param / optimizer_._phi(r)[:, None] / optimizer_._phi(c)[None, :]
                torch.testing.assert_close(
                    bare.norm(dim=1), torch.ones_like(bare[:, 0]), atol=1e-6, rtol=0
                )


@pytest.mark.parametrize("fc1,rows,cols", [(True, 48, 20), (False, 20, 24)])
def test_expert_main_grad_optimizer_mapping(tp1, fc1, rows, cols):
    check_mapping(tp1, fc1=fc1, rows=rows, cols=cols)


@pytest.mark.parametrize("fc1,rows,cols", [(True, 3840, 1792), (False, 1792, 1920)])
def test_chonk_shapes_main_grad_optimizer_mapping(tp1, fc1, rows, cols):
    check_mapping(tp1, fc1=fc1, rows=rows, cols=cols)
