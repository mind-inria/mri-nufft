"""Field map generator module."""

import numpy as np
import sklearn.cluster as sc  # FIXME Use scipy
from numpy.typing import NDArray
from collections.abc import Sequence

from mrinufft._array_compat import get_array_module, with_numpy, with_numpy_cupy
from mrinufft._utils import MethodRegister, LazyArray

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
        T2* map of shape (*shape) in [ms].
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
    base_params=r"""\
field_map : NDArray
    The field map (off-resonance map) in rad/s, If complex-valued, the real part
    is interpreted as R2* mapping. If real-valued this is the field
    inhomogeneities in Hz. and will be multiplied by :math:`2*j*\pi`
readout_time : NDArray
    The vector of time points (in seconds) at which to compute phase evolution.
mask : NDArray
    Binary mask indicating object region for field map/statistics.
L : int, optional
    Number of virtual centers or basis functions retained (default is 9).
n_bins : int, optional
    Number of histogram bins for off-resonance value clustering (default is 1000).
weights : {"full", "ones", "sqrt", "log"}, optional
    Weighting strategy for histogram: "full" (default), "ones", "sqrt", or "log".
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
""",
)


register_orc = MethodRegister("orc_factorization", _field_map_docs)
get_orc_factorization = register_orc.make_getter()


@with_numpy_cupy
def get_complex_fieldmap_rad(
    b0_map: NDArray, r2star_map: NDArray | None = None
) -> NDArray:
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
        The complex valued field-map in radian, :math:`\Delta=R_2^* + 2j\pi\Delta f`,
        in [rad/s]. This is used in the inhomogeneity term
        :math:`\exp(\Delta(\boldsymbol{r})t)\exp(2j\pi\boldsymbol{r}\boldsymbol{k}(t))`

    See Also
    --------
    :ref:`_nufft-orc`

    """
    xp = get_array_module(b0_map)
    if xp.iscomplexobj(b0_map):
        return b0_map

    field_map = 2 * np.pi * np.complex64(1j) * xp.float32(b0_map)

    if r2star_map is not None:
        r2star_map = xp.asarray(r2star_map, dtype=xp.float32)
        field_map += r2star_map

    return field_map


def _create_histogram(
    field_map: NDArray,
    mask: NDArray | None,
    n_bins: int | tuple[int, int] = 1000,
    weights: str = "full",
):
    """
    Quantize the field map in n_bins values.

    Parameters
    ----------
    field_map : NDArray

    """
    xp = get_array_module(field_map)

    z = field_map[mask].ravel().view(xp.float32)
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

    # flatten histogram values and centers
    h_counts = h_counts.ravel()
    h_centers_cpx = h_centers_cpx.ravel()

    # Change the weighting according to args
    if weights == "ones":
        h_counts = xp.array(h_counts != 0).astype(int)
    elif weights == "sqrt":
        h_counts = xp.sqrt(h_counts)
    elif weights == "log":
        h_counts = xp.log(1 + h_counts)
    elif weights != "full":
        raise NotImplementedError(f"Unknown coefficient weightning: {weights}")

    return h_counts, h_centers_cpx


@with_numpy
def _create_variable_density(centers: NDArray, counts: NDArray, L: int):
    # Compute kmeans to get custom centers
    km = sc.KMeans(n_clusters=L, random_state=0)
    km = km.fit(centers.reshape((-1, 1)), sample_weight=counts)
    centers = np.array(sorted(km.cluster_centers_)).flatten()
    return centers


@register_orc("mfi")
@with_numpy_cupy
def compute_mfi_coefficients(
    field_map: NDArray,
    readout_time: NDArray,
    mask: NDArray,
    L: int = 9,
    n_bins: int | tuple[int, int] = 1000,
    weights: str = "full",
):
    """
    Compute Model-Free Interpolation (MFI) coefficients for field map correction.

    Approximates the effect of off-resonance by fitting eigenspaces directly to
    the off-resonance phase evolution using weighted centers and spatial
    histogram of the field map. The method outputs a basis set (B),
    interpolation coefficients (C), and sampled exponentials (E) needed for
    efficient off-resonance correction and interpolation.

    Parameters
    ----------
    ${base_params}

    ${returns}

    Notes
    -----
    See also the referenced GRAPPATMI and MFI methods.
    """
    field_map = get_complex_fieldmap_rad(field_map)
    xp = get_array_module(field_map)
    # Format the input and apply the weight option
    h_k, w_k = _create_histogram(field_map, mask, n_bins, weights)
    if weights == "ones":
        w_l = xp.linspace(xp.min(field_map), xp.max(field_map), L)
    else:
        w_l = _create_variable_density(w_k, h_k, L)
    h_k = h_k.reshape((1, -1))

    # Compute B as a phase shift
    B = xp.exp(xp.outer(readout_time, w_l))
    E = xp.exp(xp.outer(readout_time, w_k))

    # Compute C with a Least Square interpolation
    C, _, _, _ = xp.linalg.lstsq(B, E, rcond=None)
    return B, C, E


@register_orc("mti_old")
@with_numpy_cupy
def compute_mti_coefficients2(
    field_map: NDArray,
    readout_time: NDArray,
    mask: NDArray,
    L: int = 9,
    n_bins: int | tuple[int, int] = 1000,
    weights: str = "full",
):
    """
    Compute Model-Time Interpolation (MTI) coefficients for field map correction.

    Approximates the effect of off-resonance by modeling the phase evolution as
    a function of time using a basis in the time domain. Constructs a low-rank
    basis for the temporal phase modulation and returns phase bases,
    interpolation matrices, and sampled exponentials.

    Parameters
    ----------
    ${base_params}
    lazy: bool, default False
        If True, don't return an explicit array for C, instead value will be
        computed lazily.

    ${returns}
    """
    # Format the input and apply the weight option
    field_map = get_complex_fieldmap_rad(field_map)
    xp = get_array_module(field_map)
    h_k, w_k = _create_histogram(field_map, mask, n_bins, weights)
    h_k = h_k.reshape((-1, 1))
    t_l = xp.linspace(xp.min(readout_time), xp.max(readout_time), L)

    # Compute C as a phase shift
    C = xp.exp(xp.outer(w_k, -t_l))
    E = xp.exp(xp.outer(w_k, -readout_time))
    Ch = xp.sqrt(h_k) * C
    Eh = xp.sqrt(h_k) * E

    # Compute B with a Least Square interpolation
    B, _, _, _ = xp.linalg.lstsq(Ch, Eh, rcond=None)
    return B.T, C.T, E.T


@register_orc("svd")
@with_numpy_cupy
def compute_svd_coefficients(
    field_map: NDArray,
    readout_time: NDArray,
    mask: NDArray,
    L: int = 9,
    n_bins: int | tuple[int, int] = 1000,
    weights: str = "full",
):
    """
    Compute off-resonance correction coefficients using SVD.

    Uses weighted SVD to extract L dominant phase basis components from the
    exponentiated phase matrix sampled at field map histogram centers. Outputs SVD basis
    (B), interpolation matrix (C), and exponential phase matrix (E).

    Parameters
    ----------
    ${base_params}

    ${returns}
    """
    # Format the input and apply the weight option

    field_map = get_complex_fieldmap_rad(field_map)
    xp = get_array_module(field_map)
    h_k, w_k = _create_histogram(field_map, mask, n_bins, weights)
    h_k = h_k.reshape((1, -1))

    # Compute B with a Singular Value Decomposition
    E = xp.exp(xp.outer(readout_time, w_k))
    u, _, _ = xp.linalg.svd(xp.sqrt(h_k) * E)
    B = u[:, :L]

    # Compute C with a Least Square interpolation
    # (Redundant with C=DV from E=UDV using L singular values
    # but it avoids 0 division issues when weighted)
    C, _, _, _ = xp.linalg.lstsq(B, E, rcond=None)
    return B, C, E


@register_orc("tsvd")
@with_numpy_cupy
def compute_tsvd_coefficients(
    field_map: NDArray,
    readout_time: NDArray,
    mask: NDArray,
    L: int = 9,
    n_bins: int | tuple[int, int] = 1000,
    weights: str = "full",
):
    """
    Compute off-resonance correction coefficients using tSVD.

    Utilizes truncated SVD to extract L leading singular vectors from the weighted
    exponentiated phase matrix sampled at field map histogram centers. Outputs tSVD
    basis (B), interpolation matrix (C), and exponential phase matrix (E).

    Parameters
    ----------
    ${base_params}

    ${returns}
    """
    field_map = get_complex_fieldmap_rad(field_map)
    xp = get_array_module(field_map)
    if xp.__name__ == "cupy":
        from cupyx.scipy.sparse.linalg import svds
    else:
        from scipy.sparse.linalg import svds
    # Format the input and apply the weight option
    h_k, w_k = _create_histogram(field_map, mask, n_bins, weights)
    h_k = h_k.reshape((1, -1))

    # Compute B with a Singular Value Decomposition
    E = xp.exp(xp.outer(readout_time, w_k))
    B, _, _ = svds(xp.sqrt(h_k) * E, L)

    # Compute C with a Least Square interpolation
    # (Redundant with C=DV from E=UDV using L singular values
    # but it avoids 0 division issues when weighted)
    C, _, _, _ = xp.linalg.lstsq(B, E, rcond=None)
    return B, C, E


@register_orc("alt_irls")
@with_numpy_cupy
def compute_alt_irls_coefficients(
    field_map, readout_time, mask, L=9, n_bins=1000, weights="full", p=2, rng=None
):
    """
    Compute off-resonance correction coefficients using IRLS.

    Alternately computes and refines two bases for off-resonance interpolation using
    IRLS (Iterative Reweighted Least Square) on the weighted exponential phase evolution
    matrix. This method can be more robust to outliers and nonlinearities by tuning the
    norm parameter `p`.

    Parameters
    ----------
    ${base_params}
    p : float, optional
        Norm index in (1, 2]. Smaller p places more emphasis on robustness (default 2).
    rng: np.random.Generator, optional
        Random number generator for initializing bases. If None, a new generator is
        created.

    ${returns}
    """
    # Format the input and apply the weight option
    field_map = get_complex_fieldmap_rad(field_map)
    xp = get_array_module(field_map)

    h_k, w_k = _create_histogram(field_map, mask, n_bins, weights)
    h_k = h_k.reshape((1, -1))
    # Compute C as a phase shift
    if not isinstance(rng, xp.random.Generator):
        rng = xp.random.default_rng(rng)
    E = xp.exp(1j * xp.outer(w_k, readout_time))
    B = (rng.random((len(readout_time), L)) + 1j * rng.random((len(readout_time), L))).T
    C = (rng.random((L, n_bins)) + 1j * rng.random((L, n_bins))).T

    # Compute B, C with Alternated Iteratively Reweighted Least Square interpolation
    for _ in range(20):
        for _ in range(5):
            error = xp.linalg.norm((E.T - B.T @ C.T), ord=1, axis=0).reshape((-1, 1))
            error = xp.clip(error, 1e-8, None)
            h_p = (error) ** (p - 2)
            B, _, _, _ = xp.linalg.lstsq(
                C * xp.sqrt(h_k * h_p), E * xp.sqrt(h_k * h_p), rcond=None
            )
        C, _, _, _ = xp.linalg.lstsq(B.T, E.T, rcond=None)
        C = C.T
    return B.T, C.T, E.T


@register_orc("mti")
@with_numpy_cupy
def compute_mti_coefficients(
    field_map: NDArray,
    readout_time: NDArray,
    L: int = 6,
    n_bins: tuple[int, int] | int = (40, 1000),
    mask: NDArray | None = None,
    weights: str = "full",
    lazy: bool = True,
):
    r"""Compute off-resonance correction coefficients using MTI.

    Parameters
    ----------
    ${base_params}
    lazy: bool, default False
        If True, use a lazy evaluation scheme for C

    ${returns}

    References
    ----------
    From Sigpy: https://github.com/mikgroup/sigpy
    and MIRT (mri_exp_approx.m): https://web.eecs.umich.edu/~fessler/code/
    """
    xp = get_array_module(field_map)

    if isinstance(n_bins, int) and xp.iscomplexobj(field_map):
        n_bins = (10, n_bins)
    elif isinstance(n_bins, Sequence) and xp.isrealobj(field_map):
        n_bins = n_bins[-1]
    # enforce data types
    readout_time = xp.asarray(readout_time, dtype=xp.float32).ravel()

    hk, zk = _create_histogram(field_map, mask, n_bins, weights)

    # generate time for each segment
    tl = xp.linspace(
        readout_time.min(), readout_time.max(), L, dtype=xp.float32
    )  # time segments centers in [s]

    # prepare for basis calculation
    E = xp.exp(xp.outer(-tl, zk))
    w = hk**0.5
    p = w * xp.linalg.pinv(w * E.T)

    # actual temporal basis calculation
    B = p @ xp.exp(xp.outer(zk, -readout_time))
    B = B.astype(xp.complex64)

    if lazy:  # C[l,*xyz]= exp(-t_[l] *field_map[*xyz])
        C = LazyArray(
            lambda idx: xp.nan_to_num(
                xp.exp(-tl[idx].item() * field_map), nan=0.0, posinf=0.0, neginf=0.0
            ),
            shape=(L, *field_map.shape),
            dtype=xp.complex64,
        )
    else:
        C = xp.exp(-tl[:, None] * field_map[None, ...])
    return B, C, E
