"""
Base Fourier Operator interface.

from https://github.com/CEA-COSMIC/pysap-mri

:author: Pierre-Antoine Comby
"""


class FourierOperatorBase:
    """Base Fourier Operator class.

    Every (Linear) Fourier operator inherits from this class,
    to ensure that we have all the functions rightly implemented
    as required by ModOpt.

    Attributes
    ----------
    shape: tuple
        The shape of the image space (in 2D or 3D)
    n_coils: int
        The number of coils.
    uses_sense: bool
        True if the operator uses sensibility maps.

    Methods
    -------
    op(data)
        The forward operation (image -> kspace)
    adj_op(coeffs)
        The adjoint operation (kspace -> image)
    """

    def __init__(self):
        self._uses_sense = False

    def op(self, data):
        """Compute operator transform.

        Parameters
        ----------
        data: np.ndarray
            input as array.

        Returns
        -------
        result: np.ndarray
            operator transform of the input.
        """
        raise NotImplementedError("'op' is an abstract method.")

    def adj_op(self, coeffs):
        """Compute adjoint operator transform.

        Parameters
        ----------
        x: np.ndarray
            input data array.

        Returns
        -------
        results: np.ndarray
            adjoint operator transform.
        """
        raise NotImplementedError("'adj_op' is an abstract method.")

    @property
    def uses_sense(self):
        """Return True if the operator uses sensitivity maps."""
        return self._uses_sense

    @property
    def shape(self):
        """Shape of the image space of the operator."""
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = tuple(shape)

    @property
    def n_coils(self):
        """Return number of coil of the image space of the operator."""
        return self._n_coils

    @n_coils.setter
    def n_coils(self, n_coils):
        if n_coils < 1 or not int(n_coils) == n_coils:
            raise ValueError(f"n_coils should be a positive integer, {type(n_coils)}")
        self._n_coils = int(n_coils)

    def with_off_resonnance_correction(self, B, C, indices):
        """Return a new operator with Off Resonnance Correction."""
        from ..off_resonnance import MRIFourierCorrected

        return MRIFourierCorrected(self, B, C, indices)
