"""Fieldmap cases we want to test."""

from mrinufft.extras import make_b0map, make_t2smap


class CasesB0maps:
    """B0 field maps cases we want to test.

    Each case return a field map and the binary spatial support of the object.
    """

    def case_real2D(self, N=64, b0range=(-300, 300)):
        """Create a real (B0 only) 2D field map."""
        return make_b0map(2 * [N])

    def case_real3D(self, N=32, b0range=(-300, 300)):
        """Create a real (B0 only) 3D field map."""
        return make_b0map(3 * [N])


class CasesZmaps:
    """Complex zmap field maps cases we want to test.

    Each case return a field map and the binary spatial support of the object.
    """

    def case_complex2D(self, N=64, b0range=(-300, 300), t2svalue=15.0):
        """Create a complex (R2* + 1j * B0) 2D field map."""
        # Generate real and imaginary parts
        t2smap, _ = make_t2smap(2 * [N])
        b0map, mask = make_b0map(2 * [N])

        # Convert T2* map to R2* map
        t2smap = t2smap * 1e-3  # ms -> s
        r2smap = 1.0 / (t2smap + 1e-9)  # Hz
        r2smap = mask * r2smap

        # Calculate complex fieldmap (Zmap)
        zmap = r2smap + 1j * b0map

        return zmap, mask

    def case_complex3D(self, N=32, b0range=(-300, 300), t2svalue=15.0):
        """Create a complex (R2* + 1j * B0) 3D field map."""
        # Generate real and imaginary parts
        t2smap, _ = make_t2smap(3 * [N])
        b0map, mask = make_b0map(3 * [N])

        # Convert T2* map to R2* map
        t2smap = t2smap * 1e-3  # ms -> s
        r2smap = 1.0 / (t2smap + 1e-9)  # Hz
        r2smap = mask * r2smap

        # Calculate complex fieldmap (Zmap)
        zmap = r2smap + 1j * b0map

        return zmap, mask
