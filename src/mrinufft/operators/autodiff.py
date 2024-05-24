"""Torch autodifferentiation for MRI-NUFFT."""

import torch


class _NUFFT_OP(torch.autograd.Function):
    """
    Autograd support for op nufft function.

    This class is implemented by an efficient approximation of Jacobian Matrices.

    References
    ----------
    Wang G, Fessler J A. "Efficient approximation of Jacobian matrices involving a
    non-uniform fast Fourier transform (NUFFT)."
    IEEE Transactions on Computational Imaging, 2023, 9: 43-54.
    """

    @staticmethod
    def forward(ctx, x, traj, nufft_op):
        """Forward image -> k-space."""
        ctx.save_for_backward(x, traj)
        ctx.nufft_op = nufft_op
        return nufft_op.op(x)

    @staticmethod
    def backward(ctx, dy):
        """Backward image -> k-space."""
        (x, traj) = ctx.saved_tensors

        im_size = x.size()[1:]
        r = [torch.linspace(-size / 2, size / 2 - 1, size) for size in im_size]
        grid_r = torch.meshgrid(*r, indexing="ij")
        grid_r = torch.stack(grid_r, dim=0).type_as(x)[None, ...]

        grid_x = x * grid_r  # Element-wise multiplication: x * r
        nufft_dx_dom = torch.cat(
            [
                ctx.nufft_op.op(grid_x[:, i : i + 1, :, :])
                for i in range(grid_x.size(1))
            ],
            dim=1,
        )  # Compute A(x * r) for each channel and concatenate along this dimension

        grad_traj = torch.transpose(
            (-1j * torch.conj(dy) * nufft_dx_dom).squeeze(), 0, 1
        ).type_as(
            traj
        )  # Compute gradient with respect to trajectory: -i * dy' * A(x * r)

        return ctx.nufft_op.adj_op(dy), grad_traj, None


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
            raise ValueError("Squeezing dimensions is not " "supported for autodiff.")
        self.nufft_op = nufft_op

    def op(self, x, traj):
        r"""Compute the forward image -> k-space."""
        return _NUFFT_OP.apply(x, traj, self.nufft_op)

    def adj_op(self, kspace):
        r"""Compute the adjoint k-space -> image."""
        return _NUFFT_ADJOP.apply(kspace, self.nufft_op)

    def __getattr__(self, name):
        """Get the attribute from the root operator."""
        return getattr(self.nufft_op, name)
