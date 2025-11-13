"""Field map Module.

This module provides methods to generate dummy B0map as well as estimation of the
bilinearization of the off-resonance model (See :ref:`nufft-orc` for a detailed
explanation).



.. autoregistry:: orc_factorization

"""

import numpy as np
from collections.abc import Callable

from numpy.typing import ArrayLike, NDArray

from mrinufft._array_compat import (
    get_array_module,
    with_numpy,
    with_numpy_cupy,
)
from mrinufft._utils import MethodRegister

#######################
# Generate Dummy Data #
#######################


def make_b0map(shape, b0range=(-300, 300), mask=None):
    """
    Make radial B0 map.

    Parameters
    ----------
    shape: tuple[int]
        Matrix size.
    b0range: tuple[float], optional
        Frequency shift range in [Hz]. The default is (-300, 300).
    mask: np.ndarray
        Spatial support of the object. If not provided,
        build a radial mask with radius ``radius=0.3*shape``

    Returns
    -------
    np.ndarray
        B0 map in Hz, values included in ``b0range``, and shape ``shape``
    np.ndarray
        Spatial support binary mask

    """
    ndim = len(shape)
    if ndim == 2:
        radial_mask, fieldmap = _make_disk(shape)
    elif ndim == 3:
        radial_mask, fieldmap = _make_sphere(shape)
    if mask is None:
        mask = radial_mask

    # build map
    fieldmap *= mask
    fieldmap = (b0range[1] - b0range[0]) * fieldmap / fieldmap.max() + b0range[0]  # Hz
    fieldmap *= mask

    # remove nan
    fieldmap = np.nan_to_num(fieldmap, neginf=0.0, posinf=0.0)

    return fieldmap.astype(np.float32), mask


def make_t2smap(shape, t2svalue=15.0, mask=None):
    """
    Make homogeneous T2* map.

    Parameters
    ----------
    shape: tuple[int]
        Matrix size.
    t2svalue: float, optional
        Object T2* in ms. The default is 15.0.
    mask: np.ndarray
        Spatial support of the object. If not provided,
        build a radial mask with ``radius = 0.3 * shape``

    Returns
    -------
    np.ndarray
        T2* map of shape ``(*shape)`` in [ms].
    np.ndarray, optional
        Spatial support binary mask.

    """
    ndim = len(shape)
    if ndim == 2:
        radial_mask, fieldmap = _make_disk(shape)
    elif ndim == 3:
        radial_mask, fieldmap = _make_sphere(shape)
    if mask is None:
        mask = radial_mask

    # build map
    fieldmap = t2svalue * mask  # ms

    # remove nan
    fieldmap = np.nan_to_num(fieldmap, neginf=0.0, posinf=0.0)

    return fieldmap.astype(np.float32), mask


def _make_disk(shape, frac_radius=0.3):
    """Make circular binary mask."""
    ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx]
    yy, xx = yy - ny // 2, xx - nx // 2
    yy, xx = yy / ny, xx / nx
    rr = (xx**2 + yy**2) ** 0.5
    return rr < frac_radius, rr


def _make_sphere(shape, frac_radius=0.3):
    """Make spherical binary mask."""
    nz, ny, nx = shape
    zz, yy, xx = np.mgrid[:nz, :ny, :nx]
    zz, yy, xx = zz - nz // 2, yy - ny // 2, xx - nx // 2
    zz, yy, xx = zz / nz, yy / ny, xx / nx
    rr = (xx**2 + yy**2 + zz**2) ** 0.5
    return rr < frac_radius, rr


########################################################
# Estimation of Off-Resonance-Correction Interpolators #
########################################################


_field_map_docs = dict(
    base_params=r"""
field_map : NDArray
    The field map (off-resonance map) in rad/s, If complex-valued, the real part
    is interpreted as R2* mapping. If real-valued this is the field
    inhomogeneities in Hz. and will be multiplied by :math:`2j\pi`
readout_time : NDArray
    The vector of time points (in seconds) at which to compute phase evolution.
mask : NDArray
    Binary mask indicating object region for field map/statistics.
L : int, optional
    Number of virtual centers or basis functions retained (default is -1,
    automatically estimated).
n_bins : int, optional
    Number of histogram bins for off-resonance value clustering (default is 1000).
lazy: bool, default False
    If True, use a lazy evaluation scheme for the space interpolator C, saving memory.
""",
    returns="""
Returns
-------
B: NDArray
    [L, nbins] phase basis matrix in the time domain.
C: NDArray
    [L, nt] interpolation matrix to transform weighted basis to phase at the
    time points; nt = len(readout_time).
E: NDArray
    [nbins, nt] exponential off-resonance phase matrix at input histogram bins.

See Also
--------
    :ref:`nufft-orc`
""",
)


register_orc = MethodRegister("orc_factorization", _field_map_docs)
get_orc_factorization = register_orc.make_getter()


@with_numpy_cupy
def get_complex_fieldmap_rad(
    b0_map: NDArray, r2star_map: NDArray | None = None
) -> NDArray[np.complex64]:
    r"""Create a complex-valued field-map in rad/s.

    Parameters
    ----------
    b0_map: NDArray
        If complex-valued, returned as is.
        If real valued, it is the off-resonance field map in Hz.
    r2star_map: NDArray, optional
        :math:`R_2^*` mapping, in Hz

    Returns
    -------
    NDArray
        The complex valued field-map in radian, :math:`-\Delta=R_2^* + 2j\pi\Delta f`,
        in [rad/s]. This is used in the inhomogeneity term
        :math:`\exp(\Delta(\boldsymbol{r})t)\exp(2j\pi\boldsymbol{r}\boldsymbol{k}(t))`

    See Also
    --------
    :ref:`nufft-orc`

    """
    xp = get_array_module(b0_map)
    if xp.iscomplexobj(b0_map):
        return b0_map

    field_map = xp.complex64(2j * xp.pi) * b0_map.astype(xp.float32, copy=False)

    if r2star_map is not None:
        r2star_map = xp.asarray(r2star_map, dtype=xp.float32)
        field_map += r2star_map

    return field_map


def _create_histogram(
    field_map: NDArray[np.complex64],
    mask: NDArray | None,
    n_bins: int | tuple[int, int] = 1000,
) -> tuple[NDArray, NDArray]:
    """
    Quantize the field map in n_bins values.

    Parameters
    ----------
    field_map : NDArray

    """
    xp = get_array_module(field_map)
    masked_field_map = field_map[mask]
    if isinstance(n_bins, int):
        deltaR = xp.max(masked_field_map.real) - xp.min(masked_field_map.real)
        deltaI = xp.max(masked_field_map.imag) - xp.min(masked_field_map.imag)
        if deltaI == 0:
            n_bins = (n_bins, 1)
        elif deltaR == 0:
            n_bins = (1, n_bins)
        else:
            n_bins_r = np.maximum(1, int(xp.around(n_bins * deltaR / deltaI)))
            n_bins_i = np.around(n_bins / n_bins_r)
            n_bins = (int(n_bins_r), int(n_bins_i))

    z = field_map[mask].view(xp.float32).reshape(-1, 2)
    # create histograms
    h_counts, h_edges = xp.histogramdd(z, bins=n_bins)
    # get center of bins for real and imaginary part
    h_centers = [e[1:] - (e[1] - e[0]) / 2 for e in h_edges]

    if len(h_centers) == 2:
        h_centers_cpx = np.add.outer(
            h_centers[0], 1j * h_centers[1], dtype=np.complex64
        )
    else:
        h_centers_cpx = np.complex64(1j) * h_centers[0]

    return h_counts, h_centers_cpx


class C_lazy:
    """A lazy version of the C interpolator.

    Instead of storing C as a ``(L, Nx, Ny, Nz)`` array, we store the quantize
    version of shape L, N_bins, and the indexes mapping each voxel of the
    complex-valued field-map to the bins values.

    When accessing the L-th interpolator, the full ``(Nx, Ny, Nz)`` associated array is
    generated.


    Parameters
    ----------
    C_sr: NDArray of shape (L, Nbinsreal, Nbinsimag)
    idxr: NDArray mapping field-map real value to real bins
    idxi: NDArray mapping field-map imag value to imag bins

    Methods
    -------
    __getitem__
        Return the l-th interpolator, with the same shape as field-map.
    """

    def __init__(
        self,
        C_sr: NDArray[np.complex64],
        idxr: ArrayLike,
        idxi: ArrayLike,
        img_shape: tuple[int, int, int],
    ):
        self.C_small = C_sr
        self.idxr = idxr
        self.idxi = idxi
        self.img_shape = img_shape

    def __len__(self) -> int:
        """Get number of interpolators."""
        return len(self.C_small)

    @property
    def shape(self):
        """Overall shape of the lazy array."""
        return (len(self), *self.img_shape)

    def __getitem__(self, i: int) -> NDArray:
        """Return the original array, with proper space indexing."""
        return self.C_small[i][self.idxr, self.idxi].reshape(self.img_shape)


@with_numpy_cupy
def _full_C(
    field_map: NDArray[np.complex64],
    C_small: NDArray,
    n_bins: tuple[int, int],
    lazy: bool = False,
):
    """
    Generate a full spatial interpolator C from a quantized version.

    If `lazy=True` uses the an array like structure to generate the
    interpolators on the fly.

    Parameters
    ----------
    field_map: NDArray
        Complex-valued field-map in rad/s.
    C_small: NDArray
        Small C matrix computed at histogram centers.
    n_bins: tuple[int, int]
        Number of bins for the real and imaginary part
    lazy: bool, default False
        If True, returns a lazy version of the C matrix.

    Returns
    -------
    NDArray if lazy = False

    C_Lazy if lazy = True
    """
    xp = get_array_module(field_map)
    fr = field_map.real.ravel()
    fi = field_map.imag.ravel()
    minr, maxr = xp.min(fr), xp.max(fi)
    mini, maxi = xp.min(fi), xp.max(fi)

    dr = (maxr - minr) / n_bins[0]
    di = (maxi - mini) / n_bins[1]
    idxr, idxi = 0, 0
    if dr != 0:
        idxr = (xp.around((fr - minr) / dr)).astype(int)
        idxr = xp.clip(idxr, 0, n_bins[0] - 1)
    if di != 0:
        idxi = (xp.around((fi - mini) / di)).astype(int)
        idxi = xp.clip(idxi, 0, n_bins[1] - 1)

    C_sr = C_small.reshape(-1, *n_bins)

    if lazy:
        C_big = C_lazy(C_sr, idxr, idxi, field_map.shape)
        return C_big
    else:

        C_big = C_sr[:, idxr, idxi]
        return xp.ascontiguousarray(C_big.reshape(-1, *field_map.shape))


def _get_svds(
    xp, partial_svd
) -> Callable[[NDArray, int], tuple[NDArray, NDArray, NDArray]]:
    if partial_svd:
        if xp.__name__ == "cupy":
            from cupyx.scipy.sparse.linalg import svds
        else:
            from scipy.sparse.linalg import svds
    else:

        def svds(x, n):
            u, s, v = xp.linalg.svd(x, compute_uv=True, full_matrices=False)
            return u[:, :n], s[:n], v[:n, :]

    return svds


@register_orc("svd")
@with_numpy_cupy
def compute_svd_coefficients(
    field_map: NDArray,
    readout_time: NDArray,
    mask: NDArray,
    L: int = -1,
    n_bins: int | tuple[int, int] = 1024,
    lazy: bool = False,
    partial_svd: bool = True,
):
    r"""
    Compute off-resonance correction coefficients using an SVD.

    Solves :math:`\arg\min_{B,C} = \|E - BC \|_{F}^2`

    In practise it utilizes truncated SVD to extract L leading singular vectors from
    the weighted exponentiated phase matrix sampled at field map histogram centers.


    Parameters
    ----------
    ${base_params}
    partial_svd: bool, default True
        If True, only compute the L components required for the estimation, not
        the full SVD.

    ${returns}

    References
    ----------
    Sutton BP, Noll DC, Fessler JA. Fast, iterative image reconstruction for MRI
    in the presence of field inhomogeneities. IEEE Trans Med Imaging. 2003
    Feb;22(2):178-88. doi: 10.1109/tmi.2002.808360. PMID: 12715994.
    """
    field_map = get_complex_fieldmap_rad(field_map)
    xp = get_array_module(field_map)
    # Format the input and apply the weight option
    h_k, w_k = _create_histogram(field_map, mask, n_bins)
    hist_shape = h_k.shape

    w_k = w_k.ravel()
    h_k = h_k.ravel()
    # Compute B with a Singular Value Decomposition
    E = xp.exp(xp.outer(readout_time, w_k))
    Ew = xp.sqrt(h_k) * E
    svds = _get_svds(xp, partial_svd)
    if L == -1:
        # Empirically: maximum frequency in the E matrix.
        L = max(
            1, int(xp.ceil(abs(w_k[-1] - w_k[0]) * xp.max(readout_time) / 2 * (xp.pi)))
        )
    B, S, D = svds(Ew, L)
    # Compute C with a Least Square interpolation
    # (Redundant with C=DV from E=BSD using L singular values
    # but more robust to histogram with 0 weights.
    C, _, _, _ = xp.linalg.lstsq(B, E, rcond=None)
    C = _full_C(field_map, C, hist_shape, lazy)
    return B, C, E


@register_orc("mti")
@with_numpy_cupy
def compute_mti_coefficients(
    field_map: NDArray,
    readout_time: NDArray,
    mask: NDArray,
    L: int = -1,
    n_bins: int | tuple[int, int] = 1024,
    lazy: bool = False,
):
    """
    Compute off-resonance correction coefficients using Mixed Time interpolator (MTI).

    Parameters
    ----------
    ${base_params}

    ${returns}

    References
    ----------
    D. C. Noll, C. H. Meyer, J. M. Pauly, D. G. Nishimura and A. Macovski, "A
    homogeneity correction method for magnetic resonance imaging with
    time-varying gradients," in IEEE Transactions on Medical Imaging, vol. 10,
    no. 4, pp. 629-637, Dec. 1991, doi: 10.1109/42.108599
    """
    field_map = get_complex_fieldmap_rad(field_map)
    xp = get_array_module(field_map)
    # Format the input and apply the weight option
    h_k, w_k = _create_histogram(field_map, mask, n_bins)
    hist_shape = h_k.shape
    h_k = h_k.ravel()
    w_k = w_k.ravel()

    if L == -1:
        # from Douglas Noll PhD Thesis
        # and XXX
        L = max(
            1, int(xp.ceil(2 * abs(w_k[-1] - w_k[0]) * xp.max(readout_time) / (xp.pi)))
        )

    t_l = xp.linspace(xp.min(readout_time), xp.max(readout_time), L, dtype=xp.float32)

    # Compute C as a phase shift
    C = xp.exp(xp.outer(t_l, w_k))
    E = xp.exp(xp.outer(readout_time, w_k))
    Ch = xp.sqrt(h_k) * C
    Eh = xp.sqrt(h_k) * E

    # Compute B with a Least Square interpolation
    B, _, _, _ = xp.linalg.lstsq(Ch.T, Eh.T, rcond=None)

    C = _full_C(field_map, C, hist_shape, lazy)
    return B.T, C, E


@with_numpy
def _get_cluster_centers(hist_locs: NDArray, counts: NDArray, L: int) -> NDArray:
    """Quantized the fine histogram to L interpolators."""
    try:
        from sklearn.cluster import KMeans
    except ImportError as err:
        raise ImportError(
            "The scikit-learn module is not available. Please install "
            "it along with the [extra] dependencies "
            "or using `pip install scikit-learn`."
        ) from err
    return (
        KMeans(n_clusters=L)
        .fit(hist_locs.view(np.float32).reshape(-1, 2), sample_weight=counts)
        .cluster_centers_.view(np.complex64)
    )


@register_orc("mfi")
@with_numpy_cupy
def compute_mfi_coefficients(
    field_map: NDArray,
    readout_time: NDArray,
    mask: NDArray,
    L: int = 9,
    n_bins: int | tuple[int, int] = 1024,
    lazy: bool = True,
):
    """
    Compute off-resonance correction coefficients using Mixed-Frequency-Interpolator.

    Parameters
    ----------
    ${base_params}

    ${returns}

    References
    ----------
    Man, L.-C., Pauly, J.M. and Macovski, A. (1997), Multifrequency
    interpolation for fast off-resonance correction. Magn. Reson. Med., 37:
    785-792. https://doi.org/10.1002/mrm.1910370523

    """
    # Format the input and apply the weight option

    field_map = get_complex_fieldmap_rad(field_map)
    xp = get_array_module(field_map)
    # Format the input and apply the weight option
    h_k, w_k = _create_histogram(field_map, mask, n_bins)
    hist_shape = h_k.shape
    h_k = h_k.ravel()
    w_k = w_k.ravel()

    if L == -1:
        # From Doug Noll PhD Thesis
        L = max(
            1, int(xp.ceil(4 * abs(w_k[-1] - w_k[0])) * xp.max(readout_time) / (xp.pi))
        )
    w_l = _get_cluster_centers(w_k, h_k, L)
    # Compute B as a phase shift

    B = xp.exp(xp.outer(readout_time, w_l))
    E = xp.exp(xp.outer(readout_time, w_k))

    # Compute C with a Least Square interpolation
    C, _, _, _ = xp.linalg.lstsq(B, E, rcond=None)
    C = _full_C(field_map, C, hist_shape, lazy)
    return B, C, E
