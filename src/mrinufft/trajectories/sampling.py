"""Sampling densities and methods."""

import numpy as np
import numpy.fft as nf
import numpy.linalg as nl
import numpy.random as nr
from tqdm.auto import tqdm

from .utils import KMAX


def sample_from_density(
    nb_samples, density, method="random", *, dim_compensation="auto"
):
    """
    Sample points based on a given density distribution.

    Parameters
    ----------
    nb_samples : int
        The number of samples to draw.
    density : np.ndarray
        An array representing the density distribution from which samples are drawn,
        normalized automatically by its sum during the call for convenience.
    method : str, optional
        The sampling method to use, either 'random' for random sampling over
        the discrete grid defined by the density or 'lloyd' for Lloyd's
        algorithm over a continuous space, by default "random".
    dim_compensation : str, bool, optional
        Whether to apply a specific dimensionality compensation introduced
        in [Cha+14]_. An exponent ``N/(N-1)`` with ``N`` the number of
        dimensions in ``density`` is applied to fix the observed
        density expectation when set to ``"auto"`` and ``method="lloyd"``.
        It is also relevant to set it to ``True`` when ``method="random"``
        and one wants to create binary masks with continuous paths between
        drawn samples.

    Returns
    -------
    np.ndarray
        An array of range-normalized sampled locations.

    Raises
    ------
    ValueError
        If ``nb_samples`` exceeds the total size of the density array or if the
        specified ``method`` is unknown.

    References
    ----------
    .. [Cha+14] Chauffert, Nicolas, Philippe Ciuciu,
       Jonas Kahn, and Pierre Weiss.
       "Variable density sampling with continuous trajectories"
       SIAM Journal on Imaging Sciences 7, no. 4 (2014): 1962-1992.
    """
    try:
        from sklearn.cluster import BisectingKMeans, KMeans
    except ImportError as err:
        raise ImportError(
            "The scikit-learn module is not available. Please install "
            "it along with the [extra] dependencies "
            "or using `pip install scikit-learn`."
        ) from err

    # Define dimension variables
    shape = np.array(density.shape)
    nb_dims = len(shape)
    max_nb_samples = np.prod(shape)
    density = density / np.sum(density)

    if nb_samples > max_nb_samples:
        raise ValueError("`nb_samples` must be lower than the size of `density`.")

    # Check for dimensionality compensation
    if isinstance(dim_compensation, str) and dim_compensation != "auto":
        raise ValueError(f"Unknown string {dim_compensation} for `dim_compensation`.")
    if (dim_compensation == "auto" and method == "lloyd") or (
        isinstance(dim_compensation, bool) and dim_compensation
    ):
        density = density ** (nb_dims / (nb_dims - 1))
        density = density / np.sum(density)

    # Sample using specified method
    rng = nr.default_rng()
    if method == "random":
        choices = rng.choice(
            np.arange(max_nb_samples),
            size=nb_samples,
            p=density.flatten(),
            replace=False,
        )
        locations = np.indices(shape).reshape((nb_dims, -1))[:, choices]
        locations = locations.T + 0.5
        locations = locations / shape[None, :]
        locations = 2 * KMAX * locations - KMAX
    elif method == "lloyd":
        kmeans = (
            KMeans(n_clusters=nb_samples)
            if nb_dims <= 2
            else BisectingKMeans(n_clusters=nb_samples)
        )
        kmeans.fit(
            np.indices(density.shape).reshape((nb_dims, -1)).T,
            sample_weight=density.flatten(),
        )
        locations = kmeans.cluster_centers_ - np.array(density.shape) / 2
        locations = KMAX * locations / np.max(np.abs(locations))
    else:
        raise ValueError(f"Unknown sampling method {method}.")
    return locations


def create_cutoff_decay_density(shape, cutoff, decay, resolution=None):
    """
    Create a density with central plateau and polynomial decay.

    Create a density composed of a central constant-value ellipsis
    defined by a cutoff ratio, followed by a polynomial decay over
    outer regions as defined in [Cha+22]_.

    Parameters
    ----------
    shape : tuple of int
        The shape of the density grid, analog to the field-of-view
        as opposed to ``resolution`` below.
    cutoff : float
        The ratio of the largest k-space dimension between 0
        and 1 within which density remains uniform and beyond which it decays.
    decay : float
        The polynomial decay in density beyond the cutoff ratio.
    resolution : np.ndarray, optional
        Resolution scaling factors for each dimension of the density grid,
        by default ``None``.

    Returns
    -------
    np.ndarray
        A density array with values decaying based on the specified
        cutoff ratio and decay rate.

    References
    ----------
    .. [Cha+22] Chaithya, G. R., Pierre Weiss, Guillaume Daval-Frérot,
       Aurélien Massire, Alexandre Vignaud, and Philippe Ciuciu.
       "Optimizing full 3D SPARKLING trajectories for high-resolution
       magnetic resonance imaging."
       IEEE Transactions on Medical Imaging 41, no. 8 (2022): 2105-2117.
    """
    shape = np.array(shape)
    nb_dims = len(shape)

    if not resolution:
        resolution = np.ones(nb_dims)

    differences = np.indices(shape).astype(float)
    for i in range(nb_dims):
        differences[i] = differences[i] + 0.5 - shape[i] / 2
        differences[i] = differences[i] / shape[i] / resolution[i]
    distances = nl.norm(differences, axis=0)

    cutoff = cutoff * np.max(differences) if cutoff else np.min(differences)
    density = np.ones(shape)
    decay_mask = np.where(distances > cutoff, True, False)

    density[decay_mask] = (cutoff / distances[decay_mask]) ** decay
    density = density / np.sum(density)
    return density


def create_polynomial_density(shape, decay, resolution=None):
    """
    Create a density with polynomial decay from the center.

    Parameters
    ----------
    shape : tuple of int
        The shape of the density grid.
    decay : float
        The exponent that controls the rate of decay for density.
    resolution : np.ndarray, optional
        Resolution scaling factors for each dimension of the density grid,
        by default None.

    Returns
    -------
    np.ndarray
        A density array with polynomial decay.
    """
    return create_cutoff_decay_density(
        shape, cutoff=0, decay=decay, resolution=resolution
    )


def create_energy_density(dataset):
    """
    Create a density based on energy in the Fourier spectrum.

    A density is made based on the average energy in the Fourier domain
    of volumes from a target image dataset.

    Parameters
    ----------
    dataset : np.ndarray
        The dataset from which to calculate the density
        based on its Fourier transform, with an expected
        shape (nb_volumes, dim_1, ..., dim_N).
        An N-dimensional Fourier transform is performed.

    Returns
    -------
    np.ndarray
        A density array derived from the mean energy in the Fourier
        domain of the input dataset.
    """
    nb_dims = len(dataset.shape) - 1
    axes = range(1, nb_dims + 1)

    kspace = nf.fftshift(nf.fftn(nf.fftshift(dataset, axes=axes), axes=axes), axes=axes)
    energy = np.mean(np.abs(kspace), axis=0)
    density = energy / np.sum(energy)
    return density


def create_chauffert_density(shape, wavelet_basis, nb_wavelet_scales, verbose=False):
    """Create a density based on Chauffert's method.

    This is a reproduction of the proposition from [CCW13]_.
    A sampling density is derived from compressed sensing
    equations to maximize guarantees of exact image recovery
    for a specified sparse wavelet domain decomposition.

    Parameters
    ----------
    shape : tuple of int
        The shape of the density grid.
    wavelet_basis : str, pywt.Wavelet
        The wavelet basis to use for wavelet decomposition, either
        as a built-in wavelet name from the PyWavelets package
        or as a custom ``pywt.Wavelet`` object.
    nb_wavelet_scales : int
        The number of wavelet scales to use in decomposition.
    verbose : bool, optional
        If ``True``, displays a progress bar. Default to ``False``.

    Returns
    -------
    np.ndarray
        A density array created based on wavelet transform coefficients.

    See Also
    --------
    pywt.wavelist : A list of wavelet decompositions available in the
        PyWavelets package used inside the function.
    pywt.Wavelet : A wavelet object accepted to generate Chauffert densities.

    References
    ----------
    .. [CCW13] Chauffert, Nicolas, Philippe Ciuciu, and Pierre Weiss.
       "Variable density compressed sensing in MRI.
       Theoretical vs heuristic sampling strategies."
       In 2013 IEEE 10th International Symposium on Biomedical Imaging,
       pp. 298-301. IEEE, 2013.
    """
    try:
        import pywt as pw
    except ImportError as err:
        raise ImportError(
            "The PyWavelets module is not available. Please install "
            "it along with the [extra] dependencies "
            "or using `pip install pywavelets`."
        ) from err

    nb_dims = len(shape)
    indices = np.indices(shape).reshape((nb_dims, -1)).T

    density = np.zeros(shape)
    unit_vector = np.zeros(shape)
    for ids in tqdm(indices, disable=not verbose):
        ids = tuple(ids)
        unit_vector[ids] = 1
        fourier_vector = nf.ifftn(unit_vector)
        coeffs = pw.wavedecn(
            fourier_vector, wavelet=wavelet_basis, level=nb_wavelet_scales
        )
        coeffs, _ = pw.coeffs_to_array(coeffs)
        density[ids] = np.max(np.abs(coeffs)) ** 2
        unit_vector[ids] = 0

    density = density / np.sum(density)
    return nf.ifftshift(density)


def create_fast_chauffert_density(shape, wavelet_basis, nb_wavelet_scales):
    """Create a density based on an approximated Chauffert method.

    This implementation is based on this
    tutorial: https://github.com/philouc/mri_acq_recon_tutorial.
    It is a fast approximation of the proposition from [CCW13]_,
    where a sampling density is derived from compressed sensing
    equations to maximize guarantees of exact image recovery
    for a specified sparse wavelet domain decomposition.

    In this approximation, the decomposition dimensions are
    considered independent and computed separately to accelerate
    the density generation.

    Parameters
    ----------
    shape : tuple of int
        The shape of the density grid.
    wavelet_basis : str, pywt.Wavelet
        The wavelet basis to use for wavelet decomposition, either
        as a built-in wavelet name from the PyWavelets package
        or as a custom ``pywt.Wavelet`` object.
    nb_wavelet_scales : int
        The number of wavelet scales to use in decomposition.

    Returns
    -------
    np.ndarray
        A density array created using a faster approximation
        based on 1D projections of the wavelet transform.

    See Also
    --------
    pywt.wavelist : A list of wavelet decompositions available in the
        PyWavelets package used inside the function.
    pywt.Wavelet : A wavelet object accepted to generate Chauffert densities.

    References
    ----------
    .. [CCW13] Chauffert, Nicolas, Philippe Ciuciu, and Pierre Weiss.
       "Variable density compressed sensing in MRI.
       Theoretical vs heuristic sampling strategies."
       In 2013 IEEE 10th International Symposium on Biomedical Imaging,
       pp. 298-301. IEEE, 2013.
    """
    try:
        import pywt as pw
    except ImportError as err:
        raise ImportError(
            "The PyWavelets module is not available. Please install "
            "it along with the [extra] dependencies "
            "or using `pip install pywavelets`."
        ) from err

    nb_dims = len(shape)

    density = np.ones(shape)
    for k, s in enumerate(shape):
        unit_vector = np.zeros(s)
        density_1d = np.zeros(s)

        for i in range(s):
            unit_vector[i] = 1
            fourier_vector = nf.ifft(unit_vector)
            coeffs = pw.wavedec(
                fourier_vector, wavelet=wavelet_basis, level=nb_wavelet_scales
            )
            coeffs, _ = pw.coeffs_to_array(coeffs)
            density_1d[i] = np.max(np.abs(coeffs)) ** 2
            unit_vector[i] = 0

        reshape = np.ones(nb_dims).astype(int)
        reshape[k] = s
        density_1d = density_1d.reshape(reshape)
        density = density * density_1d

    density = density / np.sum(density)
    return nf.ifftshift(density)
