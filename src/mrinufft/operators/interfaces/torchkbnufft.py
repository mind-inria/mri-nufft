"""Pytorch MRI Nufft Operators."""

from ..base import FourierOperatorBase

TORCH_AVAILABLE = True

try:
    import torchkbnufft as torchmri
    import torchkbnufft.functional as tkbnf
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
    """

    backend = "torch"
    available = TORCH_AVAILABLE

    def __init__(self, samples, shape, n_coils=1, density=False, smaps=None, eps=1e-6):
        super().__init__()

        self.samples = samples
        self.shape = shape
        self.n_coils = n_coils
        self.eps = eps

        if density is True:
            self.density = torchmri.calc_density_compensation_function(
                ktraj=samples, im_size=shape[2:], num_iterations= 15
            )
            self.uses_density = True
        elif density is False:
            self.density = None
            self.uses_density = False
        elif torch.is_tensor(density):
            self.density = density
            self.uses_density = True
        else:
            raise ValueError(
                "argument `density` of type" f"{type(density)} is invalid."
            )
        if smaps is None:
            self.uses_sense = False
        elif torch.is_tensor(smaps):
            self.uses_sense = True
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
        if self.uses_sense:
            data_d = data * self.smaps
        else:
            data_d = data

        kb_ob = tkbnf.KbNufft(
            im_size=self.shape[2:],
        )
        return kb_ob(image=data_d, omega=self.samples, smaps=self.smaps) 

    def adj_op(self, data):
        """
        Backward Operation.
``image`` calculated at scattered Fourier locations.
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
        adjkb_ob = tkbnf.KbNufftAdjoint(
            im_size=self.shape[2:],
        )
        img = adjkb_ob(
            data=data_d,
            omega=self.samples
        )
        return torch.sum(img * torch.conj(self.smaps), dim=0)