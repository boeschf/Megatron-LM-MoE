# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


# pylint: disable=missing-function-docstring, missing-class-docstring

import torch

from megatron.core.jit import jit_fuser
from megatron.core.utils import nvtx_decorator

import os as _os
# beverin-rocm (lab T4): BEVERIN_TRITON_SSSGLU=1 runs SSSGLU as one fused Triton kernel per
# direction (beverin_sssglu.py) instead of the eager @jit_fuser ops (TORCH_COMPILE_DISABLE=1).
_BEVERIN_TRITON = _os.environ.get("BEVERIN_TRITON_SSSGLU", "0") == "1"

###### BIAS SSSGLU FUSION/ NO AUTOGRAD ################
# SSSGLU is a gated linear unit whose gate is a shifted, scaled softsign:
#   gate(x)  = softsign(x - 1) + 0.5 = (x - 1) / (1 + |x - 1|) + 0.5   (== sssglu_act in activations.py)
#   SSSGLU(y_1, y_2) = gate(y_1) * y_2
# Like RLGLU (and unlike SwiGLU/SSGLU) the gate is NOT of the form ``x * squash(x)`` -- it is the
# gate itself, applied directly, so there is no separate SiLU-style helper. The gate derivative is
#   gate'(x) = 1 / (1 + |x - 1|)**2
# which is cheap and needs nothing beyond ``|x - 1|`` (already computed for the gate). Built exactly
# like the SwiGLU/SSGLU/RLGLU fusions (fused_bias_swiglu.py / fused_bias_ssglu.py /
# fused_bias_rlglu.py): @jit_fuser forward/backward pairs wrapped in torch.autograd.Function, since
# the gate has no cross-feature reduction. The gate math is inlined here (rather than importing
# sssglu_act) to keep the jit_fuser scripting self-contained; it must stay in sync with sssglu_act
# in activations.py.


@jit_fuser
def sssglu(y):
    """Performs SSSGLU (Shifted-Scaled-Softsign-Gated Linear Unit) activation function.

    Args:
        y (torch.Tensor): Input tensor to be split into two halves along the last dimension.

    Returns:
        torch.Tensor: Result of SSSGLU activation: gate(y1) * y2, where y1, y2 are the split
            halves and gate(x) = softsign(x - 1) + 0.5.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    u = y_1 - 1
    return (0.5 + u / (1 + torch.abs(u))) * y_2


@jit_fuser
def bias_sssglu(y, bias):
    """Performs SSSGLU activation with bias addition.

    Args:
        y (torch.Tensor): Input tensor.
        bias (torch.Tensor): Bias tensor to be added to input.

    Returns:
        torch.Tensor: Result of bias addition followed by SSSGLU activation.
    """
    y = y + bias
    return sssglu(y)


@jit_fuser
def weighted_sssglu(y, weights):
    dtype = y.dtype
    res = sssglu(y) * weights
    return res.to(dtype)


@jit_fuser
def sssglu_back(g, y):
    """Computes the gradient for the SSSGLU activation function.

    With gate(x) = softsign(x - 1) + 0.5, the gate derivative is
        gate'(x) = 1 / (1 + |x - 1|)**2.
    So d/dy1 [gate(y1) * y2] = gate'(y1) * y2 and d/dy2 [gate(y1) * y2] = gate(y1).

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    u = y_1 - 1
    denom = 1 + torch.abs(u)
    gate = 0.5 + u / denom
    gate_prime = 1.0 / (denom * denom)
    return torch.cat((g * gate_prime * y_2, g * gate), -1)


@jit_fuser
def bias_sssglu_back(g, y, bias):
    """Computes the gradient for the biased SSSGLU activation function.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.
        bias (torch.Tensor): Bias tensor that was added in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor, computed after
            applying the bias addition.
    """
    y = y + bias
    return sssglu_back(g, y)


@jit_fuser
def weighted_sssglu_back(g, y, weights):
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = sssglu_back(g * weights, y)
    # precison of w may be higher than y and g, so we need to cast g to w_dtype
    weights_grad = sssglu(y) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


class BiasSSSGLUFunction(torch.autograd.Function):
    """Custom autograd function for SSSGLU activation with bias support."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, bias, fp8_input_store, cpu_offload_input):
        """Forward pass of biased SSSGLU activation."""
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
            bias.activation_offloading = True
        ctx.save_for_backward(input_for_backward, bias)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return bias_sssglu(input, bias)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of biased SSSGLU activation."""
        input, bias = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = bias_sssglu_back(grad_output, input, bias)
        return tmp, tmp, None, None


class SSSGLUFunction(torch.autograd.Function):
    """Custom autograd function for SSSGLU activation without bias."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, fp8_input_store, cpu_offload_input):
        """Forward pass of SSSGLU activation."""
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return sssglu(input)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of SSSGLU activation."""
        input = ctx.saved_tensors[0]
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = sssglu_back(grad_output, input)
        return tmp, None, None


class WeightedSSSGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weights, fp8_input_store):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return weighted_sssglu(input, weights)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp, wgrad = weighted_sssglu_back(grad_output, input, weights)
        return tmp, wgrad, None


def bias_sssglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
    """Implementation of biased SSSGLU that handles different input shapes.

    This function reshapes the input if necessary, applies the SSSGLU activation
    (with or without bias), and restores the original shape.

    Args:
        input (torch.Tensor): Input tensor to apply SSSGLU activation.
        bias (torch.Tensor, optional): Bias tensor to be added to input. If None,
            uses the bias-free SSSGLU variant.
        fp8_input_store (bool, optional): Whether to store intermediate values in FP8 format.
            Defaults to False.

    Returns:
        torch.Tensor: Result of biased SSSGLU activation.

    Raises:
        AssertionError: If input tensor does not have 2 or 3 dimensions.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if _BEVERIN_TRITON and bias is None and not fp8_input_store and not cpu_offload_input:
        from megatron.core.fusions.beverin_sssglu import TritonSSSGLU

        output = TritonSSSGLU.apply(input)
    elif bias is not None:
        output = BiasSSSGLUFunction.apply(input, bias, fp8_input_store, cpu_offload_input)
    else:
        output = SSSGLUFunction.apply(input, fp8_input_store, cpu_offload_input)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def weighted_bias_sssglu_impl(input, bias, weights, fp8_input_store=False):
    """
    Token-wise-weighted bias sssglu fusion.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted sssglu fusion")
    elif _BEVERIN_TRITON and not fp8_input_store:
        from megatron.core.fusions.beverin_sssglu import TritonWeightedSSSGLU

        output = TritonWeightedSSSGLU.apply(input, weights)
    else:
        output = WeightedSSSGLUFunction.apply(input, weights, fp8_input_store)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
