"""MRPro linear operator interface."""

try:
    from mrpro.operators import LinearOperator
except ModuleNotFoundError as exc:
    if exc.name == "mrpro":
        raise ImportError(
            "MRPro is required; install it with 'pip install mrpro'."
        ) from exc
    raise


class MRProNUFFT(LinearOperator, adjoint_as_backward=True):
    """Expose an MRI-NUFFT operator as an MRPro linear operator."""

    def __init__(self, nufft):
        super().__init__()
        self.nufft = nufft
        self._check_density()

    def _check_density(self):
        if self.nufft.uses_density:
            raise ValueError("MRPro requires an operator without density compensation.")

    def forward(self, x):
        """Apply the forward transform."""
        self._check_density()
        if x.shape != self.nufft.img_full_shape:
            raise ValueError(f"Expected image shape {self.nufft.img_full_shape}.")
        return (self.nufft.op(x).reshape(self.nufft.ksp_full_shape),)

    def adjoint(self, x):
        """Apply the adjoint transform."""
        self._check_density()
        if x.shape != self.nufft.ksp_full_shape:
            raise ValueError(f"Expected k-space shape {self.nufft.ksp_full_shape}.")
        return (self.nufft.adj_op(x).reshape(self.nufft.img_full_shape),)
