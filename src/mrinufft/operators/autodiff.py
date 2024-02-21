"""Torch autodifferentiation for MRI-NUFFT."""

import torch


class _NUFFT_OP(torch.autograd.Function):
    def __init__(self, nufft_op):
        self.nufft_op = nufft_op

    def forward(self, x):
        """Forward image -> k-space."""
        return self.nufft_op.op(x)

    def backward(self, dy):
        """Backward image -> k-space."""
        return self.nufft_op.adj_op(dy)


class _NUFFT_ADJOP(torch.autograd.Function):

    def __init__(self, nufft_op):
        self.nufft_op = nufft_op

    def forward(self, y):
        """Forward kspace-> image."""
        return self.nufft_op.adj_op(y)

    def backward(self, dx):
        """Backward kspace-> image."""
        return self.nufft_op.op(dx)


class MRINufftAutoGrad:
    """
    Wrapper around FourierOperatorBase with support for autodifferentiation.

    Parameters
    ----------
    nufft_op: FourierOperatorBase
        MRI-nufft operator.

    """

    def __init__(self, nufft_op):
        self.nufft_op = nufft_op

        self.op = _NUFFT_OP(self.nufft_op)
        self.adj_op = _NUFFT_ADJOP(self.nufft_op)

    def __getattr__(self, name):
        """Get the attribute from the root operator."""
        return getattr(self.nufft_op, name)


def make_autograd(nufft_op, variable="data"):
    """Make a new Operator with autodiff support."""
    if variable == "data":
        return MRINufftAutoGrad(nufft_op)
    else:
        raise ValueError(f"Autodiff with respect to {variable} is not supported.")
