"""Torch autodifferentiation for MRI-NUFFT."""

import torch
import numpy as np
from .._utils import NP2TORCH


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
        ctx.save_for_backward(x)
        ctx.nufft_op = nufft_op
        return nufft_op.op(x)

    @staticmethod
    def backward(ctx, dy):
        """Backward image -> k-space."""
        x = ctx.saved_tensors[0]
        grad_data = None
        grad_traj = None
        if ctx.nufft_op._grad_wrt_data:
            grad_data = ctx.nufft_op.adj_op(dy)
        if ctx.nufft_op._grad_wrt_traj:
            im_size = x.size()[1:]
            factor = 1
            if ctx.nufft_op.backend in ["gpunufft"]:
                factor *= np.pi * 2
            r = [
                torch.linspace(-size / 2, size / 2 - 1, size) * factor
                for size in im_size
            ]
            grid_r = torch.meshgrid(*r, indexing="ij")
            grid_r = torch.stack(grid_r, dim=0).type_as(x)[:, None]
            grid_x = x * grid_r  # Element-wise multiplication: x * r

            nufft_dx_dom = torch.cat(
                [ctx.nufft_op.op(grid_x[i, ...]) for i in range(grid_x.size(0))],
                dim=0,
            )
            grad_traj = -1j * torch.conj(dy) * nufft_dx_dom
            grad_traj = torch.transpose(
                torch.sum(grad_traj, dim=1),
                0,
                1,
            ).to(NP2TORCH[ctx.nufft_op.dtype])
        return grad_data, grad_traj, None


class _NUFFT_ADJOP(torch.autograd.Function):
    """Autograd support for adj_op nufft function."""

    @staticmethod
    def forward(ctx, y, traj, nufft_op):
        """Forward kspace -> image."""
        ctx.save_for_backward(y)
        ctx.nufft_op = nufft_op
        return nufft_op.adj_op(y)

    @staticmethod
    def backward(ctx, dx):
        """Backward kspace -> image."""
        y = ctx.saved_tensors[0]
        grad_data = None
        grad_traj = None
        if ctx.nufft_op._grad_wrt_data:
            grad_data = ctx.nufft_op.op(dx)
        if ctx.nufft_op._grad_wrt_traj:
            ctx.nufft_op.toggle_grad_traj()
            im_size = dx.size()[2:]
            factor = 1
            if ctx.nufft_op.backend in ["gpunufft"]:
                factor *= np.pi * 2
            r = [
                torch.linspace(-size / 2, size / 2 - 1, size) * factor
                for size in im_size
            ]
            grid_r = torch.meshgrid(*r, indexing="ij")
            grid_r = torch.stack(grid_r, dim=0).type_as(dx)[:, None]
            grid_dx = torch.conj(dx) * grid_r
            inufft_dx_dom = torch.cat(
                [ctx.nufft_op.op(grid_dx[i, ...]) for i in range(grid_dx.size(0))],
                dim=0,
            )
            grad_traj = 1j * y * inufft_dx_dom
            grad_traj = torch.transpose(torch.sum(grad_traj, dim=1), 0, 1).to(
                NP2TORCH[ctx.nufft_op.dtype]
            )
            ctx.nufft_op.toggle_grad_traj()
        return grad_data, grad_traj, None


class MRINufftAutoGrad(torch.nn.Module):
    """
    Wraps the NUFFT operator to support torch autodiff.

    Parameters
    ----------
    nufft_op: Classic Non differentiable MRI-NUFFT operator.

    batch_size : int, default None
        Number of batches to process simultaneously.
        if Provided, the Nufft operator will support different data/smaps pairs
            in a batched mode

    """

    # TODO Future improvements may include support for varying trajs across batches.

    def __init__(self, nufft_op, wrt_data=True, wrt_traj=False, batch_size=None):
        super().__init__()
        if (wrt_data or wrt_traj) and nufft_op.squeeze_dims:
            raise ValueError("Squeezing dimensions is not supported for autodiff.")

        self.nufft_op = nufft_op
        self.nufft_op._grad_wrt_traj = wrt_traj
        if wrt_traj and self.nufft_op.backend in ["finufft", "cufinufft"]:
            self.nufft_op._make_plan_grad()
        self.nufft_op._grad_wrt_data = wrt_data
        if wrt_traj:
            # We initialize the samples as a torch tensor purely for autodiff purposes.
            # It can also be converted later to nn.Parameter, in which case it is
            # used for update also.
            self._samples_torch = torch.Tensor(self.nufft_op.samples)
            self._samples_torch.requires_grad = True
        self.batch_size = batch_size
        self.paired_batch_mode = batch_size is not None

    def op(self, x, smaps=None):
        r"""Compute the forward image -> k-space."""
        if self.paired_batch_mode:
            return self.op_batched(x, smaps)
        return _NUFFT_OP.apply(x, self.samples, self.nufft_op)

    def adj_op(self, kspace, smaps=None):
        r"""Compute the adjoint k-space -> image."""
        if self.paired_batch_mode:
            return self.adj_op_batched(kspace, smaps)
        return _NUFFT_ADJOP.apply(kspace, self.samples, self.nufft_op)

    def op_batched(self, batched_imgs, batched_smaps):
        """Compute the forward batched_imgs -> batched_kspace."""
        # Each batch element independently calls NUFFT_OP.apply(...).
        # The NUFFT operator is stored via ctx.nufft_op, which already saves both
        # kspace_loc and smaps as attributes. Since ctx is isolated per element,
        # the correct smaps are used during backpropagation.
        self._check_input_shape(imgs=batched_imgs)
        self._check_input_shape(imgs=batched_smaps)
        batched_kspace = []
        for i in range(self.batch_size):
            try:
                # update smaps for proper backward computation
                self.nufft_op.smaps = batched_smaps[i]
                batched_kspace.append(
                    _NUFFT_OP.apply(batched_imgs[i], self.samples, self.nufft_op)
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed at batch index {i+1}"
                ) from e  # For an easier debugging
        return torch.stack(batched_kspace, dim=0)

    def adj_op_batched(self, batched_kspace, batched_smaps):
        """Compute the adjoint batched_kspace -> batched_imgs."""
        # NUFFT op is saved per batch element in ctx to ensure correct backpropagation.
        self._check_input_shape(ksps=batched_kspace)
        self._check_input_shape(imgs=batched_smaps)
        batched_imgs = []
        for i in range(self.batch_size):
            try:
                self.nufft_op.smaps = batched_smaps[i]
                batched_imgs.append(
                    _NUFFT_ADJOP.apply(batched_kspace[i], self.samples, self.nufft_op)
                )
            except Exception as e:
                raise RuntimeError(f"Failed at batch index {i+1}") from e
        return torch.stack(batched_imgs, dim=0)

    @property
    def samples(self):
        """Get the samples."""
        try:
            return self._samples_torch
        except AttributeError:
            return self.nufft_op.samples

    @samples.setter
    def samples(self, value):
        self._samples_torch = value
        self.nufft_op.samples = value.detach().cpu().numpy()

    def __getattr__(self, name):
        """Forward all other attributes to the nufft_op."""
        return getattr(self.nufft_op, name)

    def _check_input_shape(self, *, imgs=None, ksps=None):
        """Validate the batch size of either image or k-space input.

        Parameters
        ----------
        imgs : np.ndarray, optional
            Image data array. If provided, its batch dimension will be validated.

        ksps : np.ndarray or object, optional
            K-space data array or compatible object. If provided, its batch
            dimension will be validated.

        Raises
        ------
        ValueError
            If the batch size of input data does not match the expected batch size.
        """
        if imgs is not None:
            if imgs.shape[0] != self.batch_size:
                raise ValueError(
                    "Image batch size mismatch:"
                    f"Got {imgs.shape[0]}, expected {self.batch_size}. "
                    f"K-space shape: {imgs.shape}"
                )
        if ksps is not None:
            if ksps.shape[0] != self.batch_size:
                raise ValueError(
                    "K-space batch size mismatch:"
                    f"Got {ksps.shape[0]}, expected {self.batch_size}. "
                    f"K-space shape: {ksps.shape}"
                )
        if imgs is None and ksps is None:
            raise ValueError("Provide either `imgs` or `ksps` as input.")
