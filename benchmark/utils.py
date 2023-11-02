"""Utility for the benchmark."""
import numpy as np
import os

AnyShape = tuple[int, ...]


def get_smaps(
    shape: AnyShape,
    n_coils: int,
    antenna: str = "birdcage",
    dtype: np.dtype = np.complex64,
    cachedir="/tmp/smaps",
) -> np.ndarray:
    """Get sensitivity maps for a specific antenna.

    Parameters
    ----------
    shape
        Volume shape
    n_coils
        number of coil in the antenna
    antenna
        name of the antenna to emulate. Only "birdcage" is currently supported.
    dtype
        return datatype for the sensitivity maps.
    """
    if antenna == "birdcage":
        try:
            os.makedirs(cachedir, exist_ok=True)
            smaps = np.load(f"{cachedir}/smaps_{n_coils}_{shape}.npy")
        except FileNotFoundError:
            smaps = _birdcage_maps((n_coils, *shape), nzz=n_coils, dtype=dtype)
            np.save(f"{cachedir}/smaps_{n_coils}_{shape}.npy", smaps)
    else:
        raise NotImplementedError


def _birdcage_maps(
    shape: AnyShape, r: float = 1.5, nzz: int = 8, dtype: np.dtype = np.complex64
) -> np.ndarray:
    """Simulate birdcage coil sensitivies.

    Parameters
    ----------
    shape
        sensitivity maps shape (nc, x,y,z)
    r
        Relative radius of birdcage.
    nzz
        number of coils per ring.
    dtype

    Returns
    -------
    np.ndarray: complex sensitivity profiles.

    References
    ----------
    https://sigpy.readthedocs.io/en/latest/_modules/sigpy/mri/sim.html
    """
    if len(shape) == 4:
        nc, nz, ny, nx = shape
    elif len(shape) == 3:
        nc, ny, nx = shape
        nz = 1
    else:
        raise ValueError("shape must be [nc, nx, ny, nz] or [nc, nx, ny]")
    c, z, y, x = np.mgrid[:nc, :nz, :ny, :nx]

    coilx = r * np.cos(c * (2 * np.pi / nzz), dtype=np.float32)
    coily = r * np.sin(c * (2 * np.pi / nzz), dtype=np.float32)
    coilz = np.floor(np.float32(c / nzz)) - 0.5 * (np.ceil(nc / nzz) - 1)
    coil_phs = np.float32(-(c + np.floor(c / nzz)) * (2 * np.pi / nzz))

    x_co = (x - nx / 2.0) / (nx / 2.0) - coilx
    y_co = (y - ny / 2.0) / (ny / 2.0) - coily
    z_co = (z - nz / 2.0) / (nz / 2.0) - coilz
    rr = (x_co**2 + y_co**2 + z_co**2) ** 0.5
    phi = np.arctan2(x_co, -y_co) + coil_phs
    out = (1 / rr) * np.exp(1j * phi)

    rss = sum(abs(out) ** 2, 0) ** 0.5
    out /= rss
    out = np.squeeze(out)
    return out.astype(dtype)
