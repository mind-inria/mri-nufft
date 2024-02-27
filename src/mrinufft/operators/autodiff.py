"""Torch autodifferentiation for MRI-NUFFT."""

import torch


class _NUFFT_OP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nufft_op):
        """Forward image -> k-space."""
        ctx.save_for_backward(x)
        ctx.nufft_op = nufft_op
        return nufft_op.op(x)

    @staticmethod
    def backward(ctx, dy):
        """Backward image -> k-space."""
        x, = ctx.saved_tensors
        return ctx.nufft_op.adj_op(dy), None


class _NUFFT_ADJOP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, nufft_op):
        """Forward kspace-> image."""
        ctx.save_for_backward(y)
        ctx.nufft_op = nufft_op
        return nufft_op.adj_op(y)

    @staticmethod
    def backward(ctx, dx):
        """Backward kspace-> image."""
        y, = ctx.saved_tensors
        return ctx.nufft_op.op(dx), None


class MRINufftAutoGrad(torch.nn.Module):
    r"""
    Wraps the NUFFT operator to support autodiff.

    Parameters
    ----------
    nufft_op: NUFFT
    """

    def __init__(self, nufft_op):
        super().__init__()
        self.nufft_op = nufft_op

    def op(self, x):
        r"""Compute the forward image -> k-space."""
        return _NUFFT_OP.apply(x, self.nufft_op)

    def adj_op(self, kspace):
        r"""Compute the adjoint k-space -> image."""
        return _NUFFT_ADJOP.apply(kspace, self.nufft_op)

    def __getattr__(self, name):
        """Get the attribute from the root operator."""
        return getattr(self.nufft_op, name)


def make_autograd(nufft_op, variable="data"):
    """Make a new Operator with autodiff support."""
    if variable == "data":
        return MRINufftAutoGrad(nufft_op)
    else:
        raise ValueError(f"Autodiff with respect to {variable} is not supported.")
