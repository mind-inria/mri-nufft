"""Pytorch MRI Nufft Operators."""

from ..base import FourierOperatorBase

TORCH_AVAILABLE = True

try:
    import torchkbnufft as torchnufft
    import torch

except ImportError:
    TORCH_AVAILABLE = False


class MRITorchKbNufft(FourierOperatorBase):
    """MRI Transform Operator using Torch NUFFT.

    Parameters
    ----------
    samples: Tensor
        The samples location of shape ``Nsamples x N_dimensions``.
        It should be C-contiguous.
    shape: tuple
        Shape of the image space.
    n_coils: int
        Number of coils.
    density: bool or Tensor
       Density compensation support.
        - If a Tensor, it will be used for the density.
        - If True, the density compensation will be automatically estimated,
          using the fixed point method.
        - If False, density compensation will not be used.
    smaps: Tensor
    squeeze_dims: bool, default True
        If True, will try to remove the singleton dimension for batch and coils.
    """

    backend = "torchkbnufft"
    available = TORCH_AVAILABLE

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        n_batchs=1,
        n_trans=1,
        smaps=None,
        eps=1e-6,
        squeeze_dims=True,
    ):
        super().__init__()

        self.samples = samples
        self.shape = shape
        self.n_coils = n_coils
        self.eps = eps
        self.squeeze_dims = squeeze_dims
        self._tkb_op = torchnufft.KbNufft(im_size=self.shape)
        self._tkb_adj_op = torchnufft.KbNufftAdjoint(im_size=self.shape)

        if density is True:
            self.density = torchnufft.calc_density_compensation_function(
                ktraj=samples, im_size=shape, num_iterations=15
            )
        elif density is False:
            self.density = None
        elif torch.is_tensor(density):
            self.density = density
        else:
            raise ValueError(
                "argument `density` of type" f"{type(density)} is invalid."
            )
        if smaps is True:
            self.smaps = smaps
        elif smaps is None:
            self.smaps = None
        elif torch.is_tensor(smaps):
            self.smaps = smaps
        else:
            raise ValueError("argument `smaps` of type" f"{type(smaps)} is invalid")

    def op(self, data):
        """Forward operation.

        Parameters
        ----------
        data: Tensor

        Returns
        -------
        Tensor
        """
        kb_ob = self._tkb_op.forward(image=data, omega=self.samples, smaps=self.smaps)
        return self._safe_squeeze(kb_ob)

    def adj_op(self, data):
        """

        Backward Operation.

        Parameters
        ----------
        data: Tensor

        Returns
        -------
        Tensor
        """
        if self.uses_density:
            data_d = data * self.density
        else:
            data_d = data

        img = self._tkb_adj_op.forward(data=data_d, omega=self.samples)
        img_comb = torch.sum(img * torch.conj(self.smaps), dim=0)
        return self._safe_squeeze(img_comb)

    def _safe_squeeze(self, arr):
        """Squeeze the first two dimensions of shape of the operator."""
        if self.squeeze_dims:
            try:
                arr = arr.squeeze(axis=1)
            except ValueError:
                pass
            try:
                arr = arr.squeeze(axis=0)
            except ValueError:
                pass
        return arr
