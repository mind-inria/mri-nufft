"""Torch autodifferentiation for MRI-NUFFT."""

from mrinufft.operators.base import FourierOperatorBase, _ToggleGradPlanMixin

from mrinufft.operators.off_resonance import MRIFourierCorrected

import torch
import numpy as np
from .._array_compat import NP2TORCH, _array_to_torch
from torch.types import Tensor
from deepinv.physics.forward import LinearPhysics


def _backward_op_data(
    nufft: FourierOperatorBase, x: Tensor, dy: Tensor
) -> None | Tensor:
    if not nufft._grad_wrt_data:
        return None
    return nufft.adj_op(dy)


def _backward_op_samples(
    nufft: FourierOperatorBase, x: Tensor, dy: Tensor
) -> None | Tensor:
    if not nufft._grad_wrt_traj:
        return None
    factor = 1
    if nufft.backend in ["gpunufft"]:
        factor *= np.pi * 2
    r = [torch.linspace(-s / 2, s / 2 - 1, s) * factor for s in nufft.shape]
    grid_r = torch.meshgrid(*r, indexing="ij")
    grid_r = torch.stack(grid_r, dim=0).type_as(x)[:, None, None]
    grid_x = x * grid_r  # Element-wise multiplication: x * r
    # compute each kspace axis dimension separately
    nufft_dx_dom = torch.cat(
        [nufft.op(grid_x[i, ...])[None, :] for i in range(grid_x.size(0))],
        dim=0,
    )
    grad_traj = -1j * torch.conj(dy) * nufft_dx_dom
    grad_traj = torch.transpose(
        torch.sum(grad_traj, dim=(1, 2)),
        0,
        1,
    ).to(NP2TORCH[nufft.dtype])
    return grad_traj


def _backward_op_field_map(
    nufft: FourierOperatorBase, x: Tensor, dy: Tensor
) -> None | Tensor:
    if not nufft._grad_wrt_field_map or not isinstance(nufft, MRIFourierCorrected):
        return None
    # Compute gradient with respect to field map
    full_readout_time = _array_to_torch(
        nufft.full_readout_time,
        device=dy.device,
    )
    tmp = x * nufft.adj_op(dy * full_readout_time)
    return tmp


##############################
# Backward for adjoint NUFFT #
##############################


def _backward_adj_data(nufft, y: Tensor, dx: Tensor) -> None | Tensor:
    if not nufft._grad_wrt_data:
        return None
    return nufft.op(dx)


def _backward_adj_samples(nufft, y: Tensor, dx: Tensor) -> None | Tensor:
    if not nufft._grad_wrt_traj:
        return None

    with nufft.grad_traj_plan():
        factor = 2 * np.pi if nufft.backend == "gpunufft" else 1
        r = [torch.linspace(-s / 2, s / 2 - 1, s) * factor for s in nufft.shape]
        grid_r = torch.meshgrid(*r, indexing="ij")
        grid_r = torch.stack(grid_r, dim=0).type_as(dx)[:, None, None]
        grid_dx = torch.conj(dx) * grid_r
        # compute each kspace axis dimension separately
        inufft_dx_dom = torch.cat(
            [nufft.op(grid_dx[i, ...])[None, :] for i in range(grid_dx.size(0))],
            dim=0,
        )
        grad_traj = 1j * y * inufft_dx_dom
        # sum over n_coil and n_batchs dimensions
        grad_traj = torch.transpose(torch.sum(grad_traj, dim=(1, 2)), 0, 1).to(
            NP2TORCH[nufft.dtype]
        )
    return grad_traj


def _backward_adj_field_map(nufft, y, dx):
    if not nufft._grad_wrt_field_map or not isinstance(nufft, MRIFourierCorrected):
        return None
    # Compute gradient with respect to field map

    full_readout_time = _array_to_torch(
        nufft.full_readout_time,
        device=y.device,
    )

    tmp = dx.conj() * nufft.adj_op(y * full_readout_time)
    return tmp


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
    def forward(ctx, x, traj, field_map, nufft_op):
        """Forward image -> k-space."""
        ctx.save_for_backward(x)
        ctx.nufft = nufft_op
        return nufft_op.op(x)

    @staticmethod
    def backward(ctx, dy):
        """Backward image -> k-space."""
        x = ctx.saved_tensors[0]
        return (
            _backward_op_data(ctx.nufft, x, dy),
            _backward_op_samples(ctx.nufft, x, dy),
            _backward_op_field_map(ctx.nufft, x, dy),
            None,
        )


class _NUFFT_ADJOP(torch.autograd.Function):
    """Autograd support for adj_op nufft function."""

    @staticmethod
    def forward(ctx, y, traj, field_map, nufft_op):
        """Forward kspace -> image."""
        ctx.save_for_backward(y)
        ctx.nufft = nufft_op
        return nufft_op.adj_op(y)

    @staticmethod
    def backward(ctx, dx):
        """Backward image -> k-space."""
        y = ctx.saved_tensors[0]
        return (
            _backward_adj_data(ctx.nufft, y, dx),
            _backward_adj_samples(ctx.nufft, y, dx),
            _backward_adj_field_map(ctx.nufft, y, dx),
            None,
        )


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
        nufft_op: FourierOperatorBase | MRIFourierCorrected,
        wrt_data: bool = True,
        wrt_traj: bool = False,
        wrt_field_map: bool = False,
        paired_batch: bool = False,
    ):
        if any((wrt_data, wrt_traj, wrt_field_map)) and nufft_op.squeeze_dims:
            raise ValueError("Squeezing dimensions is not supported for autodiff.")

        super().__init__()
        self.nufft_op = nufft_op
        self.nufft_op._grad_wrt_traj = wrt_traj
        self.nufft_op._grad_wrt_data = wrt_data
        self.nufft_op._grad_wrt_field_map = wrt_field_map
        if wrt_traj:
            if self.nufft_op.backend in ["finufft", "cufinufft"]:
                self.nufft_op._make_plan_grad()
            # We initialize the samples as a torch tensor purely for autodiff purposes.
            # It can also be converted later to nn.Parameter, in which case it is
            # used for update also.
            self._samples_torch = _array_to_torch(self.nufft_op.samples)
            self._samples_torch.requires_grad = True
        self._field_map_torch = None
        if wrt_field_map and isinstance(self.nufft_op, MRIFourierCorrected):
            self._field_map_torch = _array_to_torch(self.nufft_op.field_map)
            self._field_map_torch.requires_grad = True

        self.paired_batch = paired_batch

    def op(self, x, smaps=None, samples=None, field_map=None):
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
        if field_map is not None and not isinstance(self.nufft_op, MRIFourierCorrected):
            raise ValueError("Underlying nufft operator does not support field map.")
        if isinstance(self.nufft_op, MRIFourierCorrected) and field_map is None:
            field_map = self.field_map
        return _NUFFT_OP.apply(x, self.samples, field_map, self.nufft_op)

    def adj_op(self, kspace, smaps=None, samples=None, field_map=None):
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
        if field_map is not None and not isinstance(self.nufft_op, MRIFourierCorrected):
            raise ValueError("Underlying nufft operator does not support field map.")
        if isinstance(self.nufft_op, MRIFourierCorrected) and field_map is None:
            field_map = self.field_map

        return _NUFFT_ADJOP.apply(kspace, self.samples, field_map, self.nufft_op)

    def _op_batched(
        self,
        batched_imgs: Tensor,
        batched_smaps: Tensor | None = None,
        batched_samples: Tensor | None = None,
        batched_field_map: Tensor | None = None,
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
                    _NUFFT_OP.apply(batched_imgs[i], self.samples, None, self.nufft_op)
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
                    _NUFFT_ADJOP.apply(
                        batched_kspace[i], self.samples, None, self.nufft_op
                    )
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
    def samples(self, value: Tensor):
        """Set the samples."""
        self.update_samples(value, unsafe=False)

    def update_samples(self, new_samples: Tensor, *, unsafe: bool = False):
        """Update the samples of the underlying nufft operator.

        Parameters
        ----------
        new_samples: Tensor
            New samples of shape (n_samples, ndim)
        unsafe: bool, default False
            If True, skip input validation checks.

        Notes
        -----
        If unsafe is True, the new_samples should be a Tensor of shape
        (Nsamples, N_dimensions), with the same dtype and device as the current
        samples. The new samples should be  F-ordered (column-major) and in the
        range [-pi, pi].

        If unsafe is False, this is automatically handled.
        """
        self._samples_torch = new_samples
        self.nufft_op.update_samples(new_samples.detach(), unsafe=unsafe)

    @property
    def field_map(self):
        """Get the field map."""
        if not isinstance(self.nufft_op, MRIFourierCorrected):
            raise ValueError("Underlying nufft operator does not support field map.")
        try:
            return self._field_map_torch
        except AttributeError:
            return self.nufft_op.field_map  # will fail if not MRIFourierCorrected

    @field_map.setter
    def field_map(self, value: Tensor):
        self.update_field_map(value)

    def update_field_map(self, new_field_map: Tensor):
        """Update the field map of the underlying nufft operator.

        This also recomputes the internal interpolators.
        """
        if not isinstance(self.nufft_op, MRIFourierCorrected):
            raise ValueError("Underlying nufft operator does not support field map.")
        self._field_map_torch = new_field_map
        self.nufft_op.update_field_map(new_field_map.detach())

    def __getattr__(self, name):
        """Get attribute."""
        return getattr(self.nufft_op, name)

    def _check_input_shape(
        self,
        *,
        imgs: Tensor | None = None,
        kspace: Tensor | None = None,
        smaps: Tensor | None = None,
        samples: Tensor | None = None,
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


class DeepInvPhyNufft(LinearPhysics):
    """Expose an MRINufftAutoGrad as as DeepInv Physics Operator."""

    def __init__(self, autograd_nufft):
        if not isinstance(autograd_nufft, MRINufftAutoGrad):
            raise ValueError("autograd_nufft should be an instance of MRINufftAutoGrad")
        super().__init__()
        # since autograd_nufft is a nn.Module, we need to set it this way
        # to avoid registering it as a sub-module / parameter.
        self.__dict__["_operator"] = autograd_nufft

    def A(self, x: Tensor, **kwargs) -> Tensor:
        """Forward operation."""
        return self._operator.op(x, **kwargs)

    def A_adjoint(self, y: Tensor, **kwargs) -> Tensor:
        """Adjoint operation."""
        return self._operator.adj_op(y, **kwargs)

    def A_dagger(self, y: Tensor, **kwargs) -> Tensor:
        """Adjoint operation."""
        return self._operator.nufft_op.pinv_solver(y, **kwargs)

    def __getattr__(self, name):
        """Get attribute."""
        return getattr(self._operator, name)
