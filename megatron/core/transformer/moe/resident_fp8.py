# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Resident E4M3FNUZ SSSGLU experts for the gfx942 TE list GEMM backend."""

import os
from collections import deque
from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable


_BLOCK = 128
_GROUPED_SELECTORS = (
    "NVTE_USE_CUTLASS_GROUPED_GEMM",
    "NVTE_USE_HIPKITTENS_GROUPED_GEMM",
    "NVTE_USE_CK_GROUPED_GEMM",
)


class _NativeBackend:
    """Only the block-scaled TE list route; never call a high-precision fallback."""

    def __init__(self, device):
        if not torch.version.hip or device.type != "cuda":
            raise RuntimeError("Resident FNUZ experts require ROCm and gfx942")
        if device.index != torch.cuda.current_device():
            raise RuntimeError("Set the rank's current device before using resident FNUZ experts")
        if torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx942":
            raise RuntimeError("Resident FNUZ experts require gfx942, not a different FP8 format")
        for name in _GROUPED_SELECTORS:
            if os.environ.get(name):
                raise RuntimeError(f"Unset {name} for the resident FNUZ TE list GEMM route")

        from transformer_engine.pytorch.constants import DType, TE_DType_To_Torch
        from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm
        from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

        from megatron.core.transformer.moe.sssglu_jit import sssglu_backward, sssglu_forward

        if TE_DType_To_Torch[DType.kFloat8E4M3] != torch.float8_e4m3fnuz:
            raise RuntimeError("Resident FNUZ experts require TE's native E4M3FNUZ dtype")

        def quantizer(dim, row, col):
            # TE uses 240 / amax (FP32 RNE), clips an overflowing multiplier,
            # and uses multiplier 1 at a zero or nonfinite maximum. It stores
            # the reciprocal in FP32 and uses native SATFINITE/RNE conversion.
            # Unlike the wire codec, this has no explicit tiny-scale floor.
            # Nonfinite operands are not sanitized by the adapter.
            return Float8BlockQuantizer(
                DType.kFloat8E4M3,
                rowwise=row,
                columnwise=col,
                amax_epsilon=0.0,
                force_pow_2_scales=False,
                block_scaling_dim=dim,
            )

        self.weight = quantizer(2, True, True)
        self.row = quantizer(1, True, False)
        self.column = quantizer(1, False, True)
        self.activation = sssglu_forward
        self.activation_backward = sssglu_backward
        self._grouped = general_grouped_gemm

    @staticmethod
    def decode_row(value):
        # TE reads these exact row bytes and FP32 dequant multipliers. The BF16
        # store here is part of the reference's saved-operand boundary.
        return value.dequantize(dtype=torch.bfloat16)

    @staticmethod
    def keep_on(value, stream):
        # The wrapper is not the allocator owner; record its real GPU storages.
        for data in (
            value._rowwise_data,
            value._rowwise_scale_inv,
            value._columnwise_data,
            value._columnwise_scale_inv,
        ):
            if data is not None:
                data.record_stream(stream)

    def gemm(self, operation, a, b, out, *, layout, accumulate=False):
        if not out:
            return
        # Label native launches for the component qualification's HIP trace.
        with torch.autograd.profiler.record_function(f"resident_fnuz::{operation}"):
            self._grouped(
                a,
                b,
                out,
                [None] * len(a),
                out_dtype=out[0].dtype,
                layout=layout,
                # Image 910 forwards this to a non-optional C++ vector even
                # with separate destinations. B always has padded tokens on axis 0.
                m_splits=[b_i.shape[0] for b_i in b],
                grad=not operation.endswith("_fwd"),
                accumulate=accumulate,
                use_bias=False,
                use_split_accumulator=True,
                single_output=False,
            )


@dataclass(frozen=True)
class _Rows:
    expert: int
    start: int
    count: int

    @property
    def padded(self):
        return ((self.count + _BLOCK - 1) // _BLOCK) * _BLOCK

    def pad(self, operand):
        rows = operand.narrow(0, self.start, self.count)
        if self.padded == self.count:
            return rows.contiguous()
        out = rows.new_empty((self.padded, *rows.shape[1:]))
        out[: self.count].copy_(rows)
        out[self.count :].zero_()
        return out


@dataclass(frozen=True)
class _DelayedWgrad:
    version: tuple
    active: tuple
    fc2: tuple
    fc1: tuple
    ready: torch.cuda.Event


class ResidentFP8Experts:
    """Non-module adapter using the original expert parameters and their FP32 main grads.

    ``drain`` is a one-shot operation for the next completed dgrad, in backward
    production order, not forward order. Every differentiable forward must be
    matched by one backward and one drain (also for zero rows).
    """

    def __init__(self, fc1, fc2):
        self.fc1 = tuple(fc1)
        self.fc2 = tuple(fc2)
        self.params = self.fc1 + self.fc2
        if not self.fc1 or len(self.fc1) != len(self.fc2):
            raise ValueError("Expected one FC1 and FC2 weight per resident expert")
        if len(set(self.params)) != len(self.params):
            raise ValueError("Resident FNUZ experts do not support tied weights")
        self.width, self.ffn = self.fc2[0].shape
        if self.width % _BLOCK or self.ffn % _BLOCK:
            raise ValueError("Resident FNUZ expert widths must be multiples of 128")
        for params, shape in (
            (self.fc1, (2 * self.ffn, self.width)),
            (self.fc2, (self.width, self.ffn)),
        ):
            for param in params:
                if (
                    not isinstance(param, torch.nn.Parameter)
                    or param.shape != shape
                    or param.dtype != torch.bfloat16
                    or not param.requires_grad
                ):
                    raise ValueError("Resident FNUZ experts require trainable rank-2 BF16 weights")
                param.resident_fp8_delayed_wgrad = True

        self._backend = None
        self._version = None
        self._weights = {}
        self._weights_ready = None
        self._outstanding = 0
        self._pending = deque()
        self._last_drain = None
        self._hooks = {}

    def register_main_grad_ready(self, param, hook_fn):
        """Register the sole DDP ready callback for an original expert parameter."""
        if not any(param is p for p in self.params):
            raise ValueError("Parameter does not belong to this resident FNUZ expert module")
        if param in self._hooks:
            raise RuntimeError("Duplicate resident FNUZ main-grad ready registration")
        self._hooks[param] = hook_fn

    def clear_weight_cache(self):
        """Forget all derived weights before an in-place checkpoint restore."""
        if self._outstanding:
            raise RuntimeError("Cannot restore resident FNUZ experts with undrained backwards")
        self._version = None
        self._weights.clear()
        self._weights_ready = None

    def _weight_version(self):
        # The optimizer explicitly bumps _version for writes via p.data (which
        # otherwise bypass PyTorch's version counter). Also detect storage swaps.
        return tuple((id(p), p._version, p.data_ptr()) for p in self.params)

    @staticmethod
    def _event(stream):
        event = torch.cuda.Event()
        event.record(stream)
        return event

    def _get_weights(self, rows, stream, *, populate):
        if self._weights_ready is not None:
            stream.wait_event(self._weights_ready)
        fresh = False
        weights = []
        for group in rows:
            i = group.expert
            if i not in self._weights:
                if not populate:
                    raise RuntimeError("Missing saved resident FNUZ expert weight generation")
                self._weights[i] = (
                    self._backend.weight(self.fc1[i].detach()),
                    self._backend.weight(self.fc2[i].detach()),
                )
                fresh = True
            pair = self._weights[i]
            for value in pair:
                self._backend.keep_on(value, stream)
            weights.append(pair)
        if fresh:
            self._weights_ready = self._event(stream)
        return weights

    @staticmethod
    def _zero_tail(value, group):
        if group.padded != group.count:
            value[group.count :].zero_()
        return value

    @staticmethod
    def _join(values, groups, empty, width):
        if not values:
            return empty.new_empty((0, width))
        # An exact-size result also avoids exporting a view with padded capacity.
        return torch.cat([v[: g.count] for v, g in zip(values, groups)])

    @staticmethod
    def _destinations(groups, like, width):
        return [like.new_empty((g.padded, width)) for g in groups]

    def __call__(self, x, splits, probs):
        """Return valid, unpadded BF16 token outputs; retain FP32 router derivatives."""
        if (
            len(splits) != len(self.fc1)
            or any(n < 0 for n in splits)
            or x.ndim != 2
            or x.shape != (sum(splits), self.width)
            or x.dtype != torch.bfloat16
            or probs.shape != x.shape[:1]
            or probs.device != x.device
            or probs.dtype != torch.float32
        ):
            raise ValueError("Invalid resident FNUZ BF16 activations, FP32 probabilities or splits")
        if any(
            p.device != x.device or p.dtype != x.dtype or not p.is_contiguous()
            for p in self.params
        ):
            raise RuntimeError("Resident FNUZ BF16 expert weights must be contiguous and colocated")
        if self._backend is None:
            self._backend = _NativeBackend(x.device)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Do not capture resident FNUZ expert compute in a CUDA graph")
        groups = []
        start = 0
        for i, n in enumerate(splits):
            if n:
                groups.append(_Rows(i, start, n))
            start += n
        # The parameter is a differentiable anchor, including when inputs do
        # not need gradients. No expert .grad is produced by this function.
        return _ResidentExpertFunction.apply(
            x, probs, self.fc1[0], self, tuple(groups), torch.is_grad_enabled()
        )

    def _forward(self, ctx, x, probs, groups, training_backward):
        version = self._weight_version()
        if version != self._version:
            if self._outstanding:
                raise RuntimeError("Resident FNUZ weights changed with undrained backwards")
            self.clear_weight_cache()
            self._version = version

        stream = torch.cuda.current_stream(x.device)
        backend = self._backend
        weights = self._get_weights(groups, stream, populate=True)
        xq = [backend.row(g.pad(x)) for g in groups]
        a = self._destinations(groups, x, 2 * self.ffn)
        backend.gemm("fc1_fwd", [w[0] for w in weights], xq, a, layout="TN")
        padded_probs = [g.pad(probs).unsqueeze(-1) for g in groups]
        s = [
            backend.activation(self._zero_tail(v, g), p)
            for v, p, g in zip(a, padded_probs, groups)
        ]
        sq = [backend.row(self._zero_tail(v, g)) for v, g in zip(s, groups)]
        y = self._destinations(groups, x, self.width)
        backend.gemm("fc2_fwd", [w[1] for w in weights], sq, y, layout="TN")
        if training_backward:
            ctx.save_for_backward(probs)
            ctx.saved_rows = (xq, [backend.row(v) for v in a])
            ctx.version = version
            self._outstanding += 1
        return self._join(y, groups, x, self.width)

    @torch.no_grad()
    def _backward(self, ctx, dy):
        if ctx.saved_rows is None:
            raise RuntimeError("Resident FNUZ backward is one-shot (no retain_graph replay)")
        if ctx.version != self._weight_version():
            raise RuntimeError("Resident FNUZ weights changed between forward and backward")
        (probs,) = ctx.saved_tensors
        groups = ctx.groups
        xq, aq = ctx.saved_rows
        stream = torch.cuda.current_stream(dy.device)
        backend = self._backend
        weights = self._get_weights(groups, stream, populate=False)

        padded_dy = [g.pad(dy) for g in groups]
        ds = self._destinations(groups, dy, self.ffn)
        backend.gemm(
            "fc2_dgrad",
            [w[1] for w in weights],
            [backend.row(v) for v in padded_dy],
            ds,
            layout="NN",
        )
        dp = torch.empty_like(probs)
        da, sb_col = [], []
        for g, value, d_s in zip(groups, aq, ds):
            p = g.pad(probs).unsqueeze(-1)
            a_b = self._zero_tail(backend.decode_row(value), g)
            d_a, d_p = backend.activation_backward(self._zero_tail(d_s, g), a_b, p)
            dp[g.start : g.start + g.count].copy_(d_p[: g.count])
            da.append(self._zero_tail(d_a, g))
            # Do not use forward S: it came from FC1's unquantized BF16 A.
            sb_col.append(backend.column(self._zero_tail(backend.activation(a_b, p), g)))

        dx = self._destinations(groups, dy, self.width)
        backend.gemm(
            "fc1_dgrad", [w[0] for w in weights], [backend.row(v) for v in da], dx, layout="NN"
        )
        # Wgrad's reduction axis is expert tokens, not features. Decode the
        # original FC1 row payload before quantizing on the new axis.
        xb_col = [
            backend.column(self._zero_tail(backend.decode_row(v), g)) for v, g in zip(xq, groups)
        ]
        fc2 = sb_col, [backend.column(v) for v in padded_dy]
        fc1 = xb_col, [backend.column(v) for v in da]
        self._pending.append(
            _DelayedWgrad(
                ctx.version, tuple(g.expert for g in groups), fc2, fc1, self._event(stream)
            )
        )
        ctx.saved_rows = None
        return self._join(dx, groups, dy, self.width), dp

    @torch.no_grad()
    def drain(self):
        """Accumulate exactly one queued backward and notify DDP on this stream."""
        if not self._pending:
            raise RuntimeError("Resident FNUZ backward_dw has no completed backward to drain")
        record = self._pending[0]
        if record.version != self._weight_version():
            raise RuntimeError("Resident FNUZ weights changed before delayed wgrad drain")

        grads = {}
        for param in self.fc2 + self.fc1:
            main_grad = getattr(param, "main_grad", None)
            if (
                main_grad is None
                or main_grad.dtype != torch.float32
                or main_grad.shape != param.shape
                or main_grad.device != param.device
                or not main_grad.is_contiguous()
                or param.grad is not None
            ):
                raise RuntimeError("Resident FNUZ wgrad requires a contiguous FP32 main_grad only")
            grads[param] = main_grad

        stream = torch.cuda.current_stream(self.fc1[0].device)
        stream.wait_event(record.ready)
        if self._last_drain is not None:
            stream.wait_event(self._last_drain)
        # Consume before issuing writes: if a native launch or callback fails,
        # retrying this record could double-add partially scheduled gradients.
        self._pending.popleft()
        for operation, params, (a, b) in (
            ("fc2_wgrad", self.fc2, record.fc2),
            ("fc1_wgrad", self.fc1, record.fc1),
        ):
            for value in (*a, *b):
                self._backend.keep_on(value, stream)
            out = [grads[params[i]] for i in record.active]
            for dest in out:
                dest.record_stream(stream)
            # Native gfx942 beta=1 reads/writes FP32 D directly: no FP32
            # temporary matrix and no global gradient-accumulation fusion.
            self._backend.gemm(operation, a, b, out, layout="NT", accumulate=True)
        # Include zero-expert and all-empty records. Both FC writes and prior
        # drains are ordered before any callback can make a bucket ready.
        for param in self.fc2 + self.fc1:
            hook = self._hooks.get(param)
            if hook is not None:
                hook(param)
        self._last_drain = self._event(stream)
        self._outstanding -= 1


class _ResidentExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probs, anchor, experts, groups, training_backward):
        ctx.experts = experts
        ctx.groups = groups
        return experts._forward(ctx, x, probs, groups, training_backward)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        dx, dp = ctx.experts._backward(ctx, dy)
        return dx, dp, None, None, None, None
