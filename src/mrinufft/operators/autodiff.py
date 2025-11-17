"""Torch autodifferentiation for MRI-NUFFT."""

import torch
import numpy as np
from .._array_compat import NP2TORCH
from torch.types import Tensor


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
            factor = 1
            if ctx.nufft_op.backend in ["gpunufft"]:
                factor *= np.pi * 2
            r = [
                torch.linspace(-s / 2, s / 2 - 1, s) * factor
                for s in ctx.nufft_op.shape
            ]
            grid_r = torch.meshgrid(*r, indexing="ij")
            grid_r = torch.stack(grid_r, dim=0).type_as(x)[:, None, None]
            grid_x = x * grid_r  # Element-wise multiplication: x * r
            # compute each kspace axis dimension separately
            nufft_dx_dom = torch.cat(
                [
                    ctx.nufft_op.op(grid_x[i, ...])[None, :]
                    for i in range(grid_x.size(0))
                ],
                dim=0,
            )
            grad_traj = -1j * torch.conj(dy) * nufft_dx_dom
            grad_traj = torch.transpose(
                torch.sum(grad_traj, dim=(1, 2)),
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
            factor = 1
            if ctx.nufft_op.backend in ["gpunufft"]:
                factor *= np.pi * 2
            r = [
                torch.linspace(-s / 2, s / 2 - 1, s) * factor
                for s in ctx.nufft_op.shape
            ]
            grid_r = torch.meshgrid(*r, indexing="ij")
            grid_r = torch.stack(grid_r, dim=0).type_as(dx)[:, None, None]
            grid_dx = torch.conj(dx) * grid_r
            # compute each kspace axis dimension separately
            inufft_dx_dom = torch.cat(
                [
                    ctx.nufft_op.op(grid_dx[i, ...])[None, :]
                    for i in range(grid_dx.size(0))
                ],
                dim=0,
            )
            grad_traj = 1j * y * inufft_dx_dom
            # sum over n_coil and n_batchs dimensions
            grad_traj = torch.transpose(torch.sum(grad_traj, dim=(1, 2)), 0, 1).to(
                NP2TORCH[ctx.nufft_op.dtype]
            )
            ctx.nufft_op.toggle_grad_traj()
        return grad_data, grad_traj, None


class MRINufftAutoGrad(torch.nn.Module):
    """
    Wraps the NUFFT operator to support torch autodiff.

    Parameters
    ----------
    nufft_op: FourierOperatorBase
        Classic Non differentiable MRI-NUFFT operator.
    wrt_data: bool, default True
        If True allow auto-differentiation/backpropagation with respect to
        kspace/image data.
    wrt_traj: bool, default False
        If True allow auto-differentiation/backpropagation with respect to the
        k-space trajectory (samples locations).
    paired_batch: bool, default False
        If True, an extra dimension for batchs is considered for the data.
        Data dimensions for inputs and output will be ``(batch_size,
        nufft_op.n_batchs, nufft_op.n_coils, nufft_op.n_samples)`` for k-space
        and ``(batch_Size, nufft_op.n_batchs, 1, *nufft_op.shape)`` for image
        (with smaps support).
        The NUFFT operator will support different processing different k-space /
        image data and corresponding sensitivity map pairs. This is particularly
        useful for multi-contrast reconstructions in a batched mode

    Notes
    -----
        - Even with batch_size, the data is processed sequentially.
        - Unlike the ``n_batchs`` argument of the base nufft_operator,
          ``batch_size`` allows for batching on other data than images and
          k-space. By using :meth:`op_batched` and :mod:`adj_op_batched`. In
          this context ``n_batchs`` of the NUFFT operator can be seen
          as a number of MR echos.
    """

    def __init__(
        self,
        nufft_op,
        wrt_data: bool = True,
        wrt_traj: bool = False,
        paired_batch: bool = False,
    ):
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
        self.paired_batch = paired_batch

    def op(self, x, smaps=None, samples=None):
        r"""Compute the forward image -> k-space.

        Parameters
        ----------
        x: Tensor
            Image data of shape ``(batch_size, nufft_op.n_batchs, (1 if smaps
            else nufft_op.n_coils), *nufft_op.shape)``
        smaps: Tensor, optional
            Sensitivity maps
            with shape ``(batch_size, nufft_op.n_coils, *nufft_op.shape)``
        samples: Tensor, optional
            Samples for the batches of shape ``(batch_size, nufft_op.n_samples,
            nufft_op.ndim)``

        Returns
        -------
        Tensor:
             with shape ``(batch_size, nufft_op.n_batchs, nufft_op.n_coils,
             nufft_op.n_samples)``

        Notes
        -----
        The batch_size dimension can be omitted if paired_batch_mode is false.
        """
        if self.paired_batch:
            return self._op_batched(x, smaps, samples)
        return _NUFFT_OP.apply(x, self.samples, self.nufft_op)

    def adj_op(self, kspace, smaps=None, samples=None):
        """
        Compute the adjoint k-space -> image.

        Parameters
        ----------
        kspace: Tensor
             with shape
             ``(batch_size, nufft_op.n_batchs, nufft_op.n_coils, nufft_op.n_samples)``
        smaps: Tensor, optional
            Sensitivity maps with shape
            ``(batch_size, nufft_op.n_coils, *nufft_op.shape)``
        samples: Tensor, optional
            Samples for the batches of shape
            ``(batch_size, nufft_op.n_samples, nufft_op.ndim)``

        Returns
        -------
        Tensor:
            Image data of shape ``(batch_size, nufft_op.n_batchs,
            (1 if smaps else nufft_op.n_coils), *nufft_op.shape)``

        Notes
        -----
        The batch_size dimension can be omitted if paired_batch_mode is false.

        """
        if self.paired_batch:
            return self._adj_op_batched(kspace, smaps, samples)
        return _NUFFT_ADJOP.apply(kspace, self.samples, self.nufft_op)

    def _op_batched(
        self,
        batched_imgs: Tensor,
        batched_smaps: Tensor | None = None,
        batched_samples: Tensor | None = None,
    ) -> Tensor:
        """Compute the forward batched_imgs -> batched_kspace."""
        # Each batch element independently calls NUFFT_OP.apply(...).
        # The NUFFT operator is stored via ctx.nufft_op, which already saves both
        # kspace_loc and smaps as attributes. Since ctx is isolated per element,
        # the correct smaps are used during backpropagation.
        self._check_input_shape(
            smaps=batched_smaps, imgs=batched_imgs, samples=batched_samples
        )
        batched_kspace = []
        for i in range(len(batched_imgs)):
            try:
                if batched_smaps is not None:
                    # update smaps for proper backward computation
                    self.nufft_op.smaps = batched_smaps[i]
                if batched_samples is not None:
                    self.samples = batched_samples[i]
                batched_kspace.append(
                    _NUFFT_OP.apply(batched_imgs[i], self.samples, self.nufft_op)
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed at batch index {i}"
                ) from e  # For an easier debugging
        return torch.stack(batched_kspace, dim=0)

    def _adj_op_batched(self, batched_kspace, batched_smaps=None, batched_samples=None):
        """Compute the adjoint batched_kspace -> batched_imgs."""
        # NUFFT op is saved per batch element in ctx to ensure correct backpropagation.
        self._check_input_shape(
            smaps=batched_smaps, kspace=batched_kspace, samples=batched_samples
        )
        batched_imgs = []
        for i in range(len(batched_kspace)):
            try:
                if batched_smaps is not None:
                    self.nufft_op.smaps = batched_smaps[i]
                if batched_samples is not None:
                    # updates the nufft_op samples internally
                    self.samples = batched_samples[i]
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

    def _check_input_shape(
        self, *, imgs=None, kspace=None, smaps=None, samples=None
    ) -> bool:
        """Validate the batch size of either ops or adj_op inputs.

        Raises ValueError if any mismatch is detected, return True otherwise
        """
        # 4 arguments, so there is 6 pairwise comparison (handshakes) to make:

        if imgs is not None and smaps is not None:
            D, B, C, *XYZ = imgs.shape
            D2, C2, *XYZ2 = smaps.shape
            if D != D2 or XYZ != XYZ2 or C != 1:
                raise ValueError("Shape mismatch between smaps and image")
        if kspace is not None and smaps is not None:
            D, B, C, NS = kspace.shape
            D2, C2, *XYZ2 = smaps.shape
            if D != D2:
                raise ValueError("Shape mismatch between smaps and k-space")
        if kspace is not None and samples is not None:
            D, B, C, NS = kspace.shape
            D2, NS2, N = samples.shape
            if D != D2 or NS != NS2:
                raise ValueError("Shape mismatch between k-space and samples loc")
        if imgs is not None and samples is not None:
            D, B, C, *XYZ = imgs.shape
            D2, NS2, N = samples.shape
            if D != D2 or N != len(XYZ):
                raise ValueError("Shape mismatch between samples loc and image")
        if samples is not None and smaps is not None:
            D, NS, N = samples.shape
            D2, C2, *XYZ2 = smaps.shape
            if D != D2 or N != len(XYZ2):
                raise ValueError("Shape mismatch between samples loc and smaps")
        if imgs is not None and kspace is not None:
            raise ValueError(
                "Input shape should not compare batched_img and batched_kspace"
            )

        return True
