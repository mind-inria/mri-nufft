"""DeepInverse physics interface."""

from deepinv.physics.forward import LinearPhysics
from torch.types import Tensor

from .autodiff import MRINufftAutoGrad, complex_view_wrapper


class DeepInvPhyNufft(LinearPhysics):
    """Expose an MRINufftAutoGrad as a DeepInv Physics Operator."""

    def __init__(self, autograd_nufft, viewed_as_real=False):
        if not isinstance(autograd_nufft, MRINufftAutoGrad):
            raise ValueError("autograd_nufft should be an instance of MRINufftAutoGrad")
        super().__init__()
        # since autograd_nufft is a nn.Module, we need to set it this way
        # to avoid registering it as a sub-module / parameter.
        self.__dict__["_operator"] = autograd_nufft
        self.viewed_as_real = viewed_as_real

    @complex_view_wrapper(in_space="img", out_space="ksp")
    def A(self, x: Tensor, **kwargs) -> Tensor:
        """Forward operation."""
        return self._operator.op(x, **kwargs)

    @complex_view_wrapper(in_space="ksp", out_space="img")
    def A_adjoint(self, y: Tensor, **kwargs) -> Tensor:
        """Adjoint operation."""
        return self._operator.adj_op(y, **kwargs)

    @complex_view_wrapper(in_space="ksp", out_space="img")
    def A_dagger(self, y: Tensor, **kwargs) -> Tensor:
        """Pseudo-inverse operation."""
        return self._operator.nufft_op.pinv_solver(y, **kwargs)

    def __getattr__(self, name):
        """Get attribute."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            operator = super().__getattr__("_operator")
            return getattr(operator, name)
