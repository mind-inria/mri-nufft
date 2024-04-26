"""Torch autodifferentiation for MRI-NUFFT."""

import torch


class _NUFFT_OP(torch.autograd.Function):
    """Autograd support for op nufft function."""

    @staticmethod
    def forward(ctx, x, nufft_op):
        """Forward image -> k-space."""
        ctx.save_for_backward(x)
        ctx.nufft_op = nufft_op
        return nufft_op.op(x)

    @staticmethod
    def backward(ctx, dy):
        """Backward image -> k-space."""
        (x,) = ctx.saved_tensors
        return ctx.nufft_op.adj_op(dy), None


class _NUFFT_ADJOP(torch.autograd.Function):
    """Autograd support for adj_op nufft function."""

    @staticmethod
    def forward(ctx, y, nufft_op):
        """Forward kspace -> image."""
        ctx.save_for_backward(y)
        ctx.nufft_op = nufft_op
        return nufft_op.adj_op(y)

    @staticmethod
    def backward(ctx, dx):
        """Backward kspace -> image."""
        (y,) = ctx.saved_tensors
        return ctx.nufft_op.op(dx), None


class MRINufftAutoGrad(torch.nn.Module):
    """
    Wraps the NUFFT operator to support torch autodiff.

    Parameters
    ----------
    nufft_op: Classic Non differentiable MRI-NUFFT operator.
    """

    def __init__(self, nufft_op):
        super().__init__()
        if nufft_op.squeeze_dims:
            raise ValueError("Squeezing dimensions is not "
                             "supported for autodiff.")
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
