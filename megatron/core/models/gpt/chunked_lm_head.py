"""Training-only TP1 output projection and full-vocabulary loss with bounded logits."""

import torch
from torch.autograd.function import once_differentiable


_MAX_TOKENS = 512
_WGRAD_VOCAB_TILE = 8192


class _ChunkedLMHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, loss_fn, rows):
        ctx.save_for_backward(hidden, weight, labels)
        ctx.loss_fn = loss_fn
        ctx.rows = rows
        batch, length = labels.shape
        losses = torch.empty((batch, length), dtype=torch.float32, device=hidden.device)
        for start in range(0, length, rows):
            end = min(start + rows, length)
            logits = hidden[start:end].matmul(weight.t())
            losses[:, start:end].copy_(loss_fn(labels[:, start:end], logits))
            del logits
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        hidden, weight, labels = ctx.saved_tensors
        need_dx, need_dw = ctx.needs_input_grad[:2]
        dx = torch.empty_like(hidden) if need_dx else None
        # Only the running weight gradient is full-vocabulary. Per-tile BF16 GEMMs
        # round their partial sums before the FP32 addition; qualify this against
        # the unsplit BF16 projection backward rather than claiming bitwise equality.
        dw = torch.zeros_like(weight, dtype=torch.float32) if need_dw else None
        for start in range(0, hidden.shape[0], ctx.rows):
            end = min(start + ctx.rows, hidden.shape[0])
            x = hidden[start:end]
            logits = x.matmul(weight.t()).requires_grad_(True)
            with torch.enable_grad():
                loss = ctx.loss_fn(labels[:, start:end], logits)
                (dlogits,) = torch.autograd.grad(loss, logits, grad_loss[:, start:end])
            del logits, loss
            dlogits = dlogits.flatten(0, 1)
            if need_dx:
                torch.mm(dlogits, weight, out=dx[start:end].view(-1, hidden.shape[-1]))
            if need_dw:
                x = x.flatten(0, 1)
                for lo in range(0, weight.shape[0], _WGRAD_VOCAB_TILE):
                    hi = min(lo + _WGRAD_VOCAB_TILE, weight.shape[0])
                    dw[lo:hi].add_(dlogits[:, lo:hi].t().matmul(x))
            del dlogits
        # Return gradients through the original Parameter's autograd/DDP path.
        return dx, dw.to(weight.dtype) if need_dw else None, None, None, None


def chunked_lm_head_loss(hidden, weight, labels, loss_fn, max_tokens):
    """Compute the existing per-token loss in sequence slices, recomputing in backward."""
    if (
        hidden.ndim != 3
        or weight.ndim != 2
        or labels.ndim != 2
        or hidden.shape[:2] != labels.t().shape
        or hidden.shape[-1] != weight.shape[-1]
        or hidden.shape[0] == 0
        or not 0 < hidden.shape[1] <= max_tokens <= _MAX_TOKENS
        or not hidden.is_contiguous()
        or not weight.is_contiguous()
        or hidden.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or labels.dtype != torch.long
        or hidden.device != weight.device
        or hidden.device != labels.device
    ):
        raise ValueError("Chunked LM head requires contiguous BF16 [S,B,H] and [V,H], and [B,S] labels")
    return _ChunkedLMHead.apply(hidden, weight, labels, loss_fn, max_tokens // hidden.shape[1])
