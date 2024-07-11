"""Torch autodifferentiation for MRI-NUFFT."""

import torch
import numpy as np


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
        if nufft_op._grad_wrt_data and not nufft_op._grad_wrt_traj:
            ctx.save_for_backward(x)
        elif not nufft_op._grad_wrt_data and nufft_op._grad_wrt_traj:
            ctx.save_for_backward(traj)
        else:
            ctx.save_for_backward(x, traj)
        ctx.nufft_op = nufft_op
        return nufft_op.op(x)

    @staticmethod
    def backward(ctx, dy):
        """Backward image -> k-space."""
        if ctx.nufft_op._grad_wrt_data and not ctx.nufft_op._grad_wrt_traj:
            x = ctx.saved_tensors
        elif not ctx.nufft_op._grad_wrt_data and ctx.nufft_op._grad_wrt_traj:
            traj = ctx.saved_tensors
        else:
            (x, traj) = ctx.saved_tensors
        grad_data = None
        grad_traj = None
        if ctx.nufft_op._grad_wrt_data:
            grad_data = ctx.nufft_op.adj_op(dy)
        if ctx.nufft_op._grad_wrt_traj:
            im_size = x.size()[1:]
            factor = 1
            if ctx.nufft_op.backend in ["gpunufft", "finufft"]:
                factor *= np.pi * 2
            r = [
                torch.linspace(-size / 2, size / 2 - 1, size) * factor
                for size in im_size
            ]
            grid_r = torch.meshgrid(*r, indexing="ij")
            grid_r = torch.stack(grid_r, dim=0).type_as(x)[None, ...]
            grid_x = x * grid_r  # Element-wise multiplication: x * r

            nufft_dx_dom = torch.cat(
                [ctx.nufft_op.op(grid_x[:, i, :, :]) for i in range(grid_x.size(1))],
                dim=0,
            )
            grad_traj = torch.mean(
                torch.cat(
                    [
                        torch.transpose(
                            (-1j * torch.conj(dy[:, i, :]) * nufft_dx_dom[:, i, :]),
                            0,
                            1,
                        )[None, ...]
                        for i in range(dy.shape[1])
                    ],
                    dim=0,
                ),
                dim=0,
            ).type_as(traj)
        return grad_data, grad_traj, None


class _NUFFT_ADJOP(torch.autograd.Function):
    """Autograd support for adj_op nufft function."""

    @staticmethod
    def forward(ctx, y, traj, nufft_op):
        """Forward kspace -> image."""
        if nufft_op._grad_wrt_data and not nufft_op._grad_wrt_traj:
            ctx.save_for_backward(y)
        elif not nufft_op._grad_wrt_data and nufft_op._grad_wrt_traj:
            ctx.save_for_backward(traj)
        else:
            ctx.save_for_backward(y, traj)
        ctx.nufft_op = nufft_op
        return nufft_op.adj_op(y)

    @staticmethod
    def backward(ctx, dx):
        """Backward kspace -> image."""
        if ctx.nufft_op._grad_wrt_data and not ctx.nufft_op._grad_wrt_traj:
            y = ctx.saved_tensors
        elif not ctx.nufft_op._grad_wrt_data and ctx.nufft_op._grad_wrt_traj:
            traj = ctx.saved_tensors
        else:
            (y, traj) = ctx.saved_tensors
        grad_data = None
        grad_traj = None
        if ctx.nufft_op._grad_wrt_data:
            grad_data = ctx.nufft_op.op(dx)
        if ctx.nufft_op._grad_wrt_traj:
            ctx.nufft_op.raw_op.toggle_grad_traj()
            im_size = dx.size()[2:]
            factor = 1
            if ctx.nufft_op.backend in ["gpunufft", "finufft"]:
                factor *= np.pi * 2
            r = [
                torch.linspace(-size / 2, size / 2 - 1, size) * factor
                for size in im_size
            ]
            grid_r = torch.meshgrid(*r, indexing="ij")
            grid_r = torch.stack(grid_r, dim=0).type_as(dx)[None, ...]
            grid_dx = torch.conj(dx) * grid_r
            inufft_dx_dom = torch.cat(
                [ctx.nufft_op.op(grid_dx[:, i, :, :]) for i in range(grid_dx.size(1))],
                dim=1,
            ).squeeze()
            inufft_dx_dom = inufft_dx_dom.reshape(y.shape[0], -1, y.shape[-1])
            grad_traj = torch.mean(
                torch.cat(
                    [
                        torch.transpose((1j * y[i] * inufft_dx_dom[i]), 0, 1)[None, ...]
                        for i in range(y.shape[0])
                    ],
                    dim=0,
                ),
                dim=0,
            ).type_as(traj)
            ctx.nufft_op.raw_op.toggle_grad_traj()
        return grad_data, grad_traj, None


class MRINufftAutoGrad(torch.nn.Module):
    """
    Wraps the NUFFT operator to support torch autodiff.

    Parameters
    ----------
    nufft_op: Classic Non differentiable MRI-NUFFT operator.
    """

    def __init__(self, nufft_op, wrt_data=True, wrt_traj=False):
        super().__init__()
        if (wrt_data or wrt_traj) and nufft_op.squeeze_dims:
            raise ValueError("Squeezing dimensions is not supported for autodiff.")

        self.nufft_op = nufft_op
        self.nufft_op._grad_wrt_traj = wrt_traj
        if wrt_traj and self.nufft_op.backend in ["finufft", "cufinufft"]:
            self.nufft_op.raw_op._make_plan_grad()
        self.nufft_op._grad_wrt_data = wrt_data

    def op(self, x):
        r"""Compute the forward image -> k-space."""
        return _NUFFT_OP.apply(x, self.samples, self.nufft_op)

    def adj_op(self, kspace):
        r"""Compute the adjoint k-space -> image."""
        return _NUFFT_ADJOP.apply(kspace, self.samples, self.nufft_op)

    def __getattr__(self, name):
        """Get the attribute from the root operator."""
        return getattr(self.nufft_op, name)
