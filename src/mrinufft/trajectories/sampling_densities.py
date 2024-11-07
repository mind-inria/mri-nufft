"""Sampling densities and methods."""

import numpy as np
import numpy.fft as nf
import numpy.linalg as nl
import numpy.random as nr
import pywt as pw
from sklearn.cluster import BisectingKMeans, KMeans
from tqdm.auto import tqdm

from .utils import KMAX


def sample_from_density(nb_samples, density, method="random"):
    rng = nr.default_rng()

    density = density / np.sum(density)
    shape = np.array(density.shape)
    nb_dims = len(shape)
    max_nb_samples = np.prod(shape)

    if nb_samples > max_nb_samples:
        raise ValueError("`nb_samples` must be lower than the size of `density`.")

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
    shape = np.array(shape)
    nb_dims = len(shape)

    if not resolution:
        resolution = np.ones(nb_dims)

    distances = np.indices(shape).astype(float)
    for i in range(nb_dims):
        distances[i] = distances[i] + 0.5 - shape[i] / 2
        distances[i] = distances[i] / shape[i] * resolution[i]
    distances = nl.norm(distances, axis=0)

    cutoff = cutoff * np.max(distances) if cutoff else np.min(distances)
    density = np.ones(shape)
    decay_mask = np.where(distances > cutoff, True, False)

    density[decay_mask] = (cutoff / distances[decay_mask]) ** decay
    density = density / np.sum(density)
    return density


def create_polynomial_density(shape, decay, resolution=None):
    return create_cutoff_decay_density(
        shape, cutoff=0, decay=decay, resolution=resolution
    )


def create_energy_density(dataset):
    nb_dims = len(dataset.shape) - 1
    axes = range(1, nb_dims + 1)

    kspace = nf.fftshift(nf.fftn(nf.fftshift(dataset, axes=axes), axes=axes), axes=axes)
    energy = np.mean(np.abs(kspace), axis=0)
    density = energy / np.sum(energy)
    return density


def create_chauffert_density(shape, wavelet_basis, nb_wavelet_scales, verbose=False):
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
