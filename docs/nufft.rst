.. include:: <isonum.txt>
.. __NUFFT:
====================
 The NUFFT Operator
====================

This document gives a general overview of the Non-Uniform Fast fourier transform (NUFFT), and its application to MRI.


The Non-Uniform Discrete Fourier Transform
==========================================

The Non-Uniform Discrete Fourier Transform (NUDFT) is a generalization of the Discrete Fourier Transform (DFT) to non-uniform sampling.

For a signal :math:`x` sampled at location :math:`p_0, p_1, \ldots, p_{N-1}` we want to get the frequency points (non uniformly spaced) at :math:`\nu_0, \nu_1, \ldots, \nu_{M-1}`

The 1D-NUDFT [1]_ is defined as:

.. math::

    X_k = \sum_{n=0}^{N-1} x(p_n) e^{ -2\imath \pi p_n \nu_k}

where :math:`X_k` is the frequency point at :math:`\nu_k`.

The multidimensional cases are derived by using vectorized location e.g :math:`\boldsymbol{p}_n` and  :math:`\boldsymbol{\nu_k}` as for the classical DFT.

There exist 3 types of NUDFT:

* Type 1: :math:`p_n = n/N` and :math:`\nu_k` are non-uniformly spaced
* Type 2: :math:`p_n` are non uniformly spaced and :math:`\nu_k = k/M`
* Type 3: :math:`p_n` and :math:`\nu_k` are non uniformly spaced
* If  :math:`p_n=n/N` and :math:`\nu_k=k/M` then the NUDFT is simply the Discrete Fourier Transform.

The naive implementation of the NUDFT is :math:`O(NM)`. This becomes problematic for large N and M. The NUFFT is a fast algorithm to compute the NUDFT. The NUFFT is a generalization of the Fast Fourier Transform (FFT) to non-uniform sampling. The underlying principles of the NUFFT algorithm are described briefly :ref:`down in the document <nufft-algo>`


Application in MRI
==================

In Magnetic Resonance Imaging (MRI) the raw data is acquired in the k-space, ideally corresponding to the Fourier domain.
Traditional sampling schemes of the k-space usually consist of acquired lines in a specific direction, in a Cartesian fashion.

In order to accelerate the acquisition of k-space data, one may use a non-Cartesian (i.e. non-uniformly distributed in *every* direction) sampling scheme as the latter offers increased sampling efficiency, i.e. broader k-space coverage in a given time period.

The acquisition model is usually described as:

.. math::

   y(\boldsymbol{\nu}_i) = \int_{\mathbb{R}^d} x(\boldsymbol{u}) e^{-2\imath\pi \boldsymbol{u} \cdot \boldsymbol{\nu_i}} d\boldsymbol{u} + n_i, \quad i=1,\dots,M

Where:

- :math:`x(\boldsymbol{u})` is the spatially varying image contrast acquired.
- :math:`y_1, \dots, y_M` are the sampled points at frequency locations :math:`\Omega=\lbrace \boldsymbol{\nu}_1, \dots, \boldsymbol{\nu}_M \in [-1/2, 1/2]^d\rbrace`.
  Typically images (:math:`d=2`) or volumes (:math:`d=3`) are acquired.
- :math:`n_i` is a zero-mean complex-valued Gaussian Noise, modeling the "thermal noise" of the scanner.


In practice the equation above is discretized, and the integral is replaced by a sum, using a finite number of samples :math:`N`:

.. math::
    y(\boldsymbol{\nu}_i) = \sum_{j=1}^N x(\boldsymbol{u}_j) e^{-2\imath\pi\boldsymbol{u}_j\cdot\boldsymbol{\nu_i}} + n_i, \quad i=1,\dots,M

Where :math:`\boldsymbol{u}_j` are the :math:`N` spatial locations of image voxels.
This is stated using the operator notation:

.. math::
    \boldsymbol{y} = \mathcal{F}_\Omega (\boldsymbol{x}) + \boldsymbol{n}

As the sampling locations :math:`\Omega` are non-uniform and the image locations :math:`\boldsymbol{u}_j` are uniform, :math:`\mathcal{F}_\Omega` is a NUDFT operator, and the equation above describe a Type 2 NUDFT. Similarly the adjoint operator is:



.. table:: Correspondance Table between NUFFT and MRI acquisition model.
    :widths: 25 25 25 25

    ==========  =========  ===================  ============================
    NUFFT Type  Operation  MRI Transform        Operator
    ==========  =========  ===================  ============================
    Type 1      Adjoint    Kspace |rarr| Image  :math:`\mathcal{F}_\Omega^*`
    Type 2      Forward    Image |rarr| Kspace  :math:`\mathcal{F}_\Omega`
    ==========  =========  ===================  ============================


.. attention::

   In order to reconstruct :math:`x` from :math:`y`, one has to solve the *inverse problem*, stated usually as

   .. math::
      \hat{x} = \arg\min_x \frac{1}{2} \|\mathcal{F}_\Omega(\boldsymbol{x}) - \boldsymbol{y}\|_2^2 + g(\boldsymbol{x}).

   This package focuses solely on computing :math:`\mathcal{F}_\Omega\boldsymbol{x}` or :math:`\mathcal{F}_\Omega^**\boldsymbol{y}`.
   solving this problem is **not** addressed here, but you can check `pysap-mri <https://github.com/CEA-COSMIC/pysap-mri>`_ for this purpose.

Extension of the Acquisition model
----------------------------------
The MRI acquisition model can be extended in two main ways. First by taking into account Parallel Imaging, where multiple coils are receiving data, each with a dedicated sensitivity profile.

Parallel Imaging Model
~~~~~~~~~~~~~~~~~~~~~~

In MRI the acquired signal can be received by multiple antennas ("coils"). Each coil possesses a specific sensitivity profile (i.e. each sees the object differently due to its physical layout).

The acquisition model for parallel imaging with :math:`L` coils  is:

.. math::

   y_{i,\ell} = \int_{\mathbb{R}^d} S_\ell(\boldsymbol{u})x(\boldsymbol{u}) e^{-2i\pi \boldsymbol{u} \cdot \boldsymbol{\nu_i}} d\boldsymbol{u} + n_{i,\ell}

Or using the operator notation:

.. math::

   \tilde{\boldsymbol{y}} = \begin{bmatrix}
    \mathcal{F}_\Omega S_1 \\
    \vdots  \\
    \mathcal{F}_\Omega S_L \\
    \end{bmatrix}
    \boldsymbol{x} + \boldsymbol{n}_\ell  = \mathcal{F}_\Omega S \otimes \boldsymbol{x} + \tilde{\boldsymbol{n}}

Where :math:`S_1, \dots, S_L` are the sensitivity maps of each coil. Such sensitivity maps can be acquired separetely by acquiring low frequency of the kspace, or estimated from the data.

..
    TODO Add ref to SENSE and CG-Sense

Off-resonance correction model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The constant magnetic field applied in a MRI machine :math:`B0` (with a typical intensity 1.5, 3 or 7 Tesla) is inherently disturbed at tissue interfaces with owing to different magnetic susceptibilities (such as air-tissue interfaces in the nose and ear canals). Those field perturbations introduce a spatially varying phase shift in the frequencies acquired (noted :math:`\Delta\omega_0`), making the acquisition model deviating from the convenient Fourier model. Fortunately, this inhomogeneity map can be acquired separatly or estimated (but this goes beyond the scope of this package) and integrated in the model as:

.. math::

   y(t_i) = \int_{\mathbb{R}^d} x(\boldsymbol{u}) e^{-2\imath\pi \boldsymbol{u} \cdot\boldsymbol{\nu_i} + \Delta\omega(\boldsymbol{u}) t_i} d\boldsymbol{u}

where :math:`t_i` is the time at which the frequency :math:`\nu_i` is acquired. Similarly at the reconstruction we have

.. math::

   x(\boldsymbol{u_n}) = \sum_{m}^M y(t_m) e^{2\imath\pi \boldsymbol{u} \cdot \boldsymbol{\nu_i}} e^{i\Delta\omega(\boldsymbol{u_n}) t_m}

With these mixed-domain field pertubations, the Fourier model does not hold anymore and the FFT algorithm cannot be used any longer.
The main approach (initially proposed by Noll et al. [2]_) is to approximate the mixed-domain exponential term by splitting it into single-domain weights :math:`b_{m, \ell}` and :math:`c_{\ell, n}`:

.. math::

   e^{i\Delta\omega(\boldsymbol{u_n}) t_m} = \sum_{\ell=1}^L b_{m, \ell}\, c_{\ell, n}

Yielding the following model, where :math:`L \ll M, N` regular Fourier transforms are performed to approximate the non-Fourier transform.

.. math::

   x(\boldsymbol{u_n}) = \sum_{\ell=1}^L c_{\ell, n} \sum_{m}^M y(t_m) b_{m, \ell} e^{2\imath\pi \boldsymbol{u} \cdot \boldsymbol{\nu_i}}

The coefficients :math:`B=(b_{m, \ell}) \in \mathbb{C}^{M\times L}` and :math:`C=(c_\ell, n) \in \mathbb{C}^{L\times N}` can be (optimally) estimated for any given :math:`L` by solving the following matrix factorisation problem [3]_:

.. math::

   \hat{B}, \hat{C} = \arg\min_{B,C} \| E- BC\|_{fro}^2

Where :math:`E_mn = e^i\Delta\omega_0(u_n)t_m`.


.. TODO Add Reference to the Code doing this.
.. TODO Reference for SVI, MTI, MFI and pointers to pysap-mri for their estimation.


.. _nufft-algo:

Subspace Projection Model
~~~~~~~~~~~~~~~~~~~~~~~~~
In several MRI applications, such as dynamic or quantitative MRI, a single acquisition provides a stack of two- or three-dimensional images, each representing a single time frame (for dynamic MRI) or a single contrast (for quantitative MRI).
To achieve a clinically feasible scan time, each frame or contrast is acquired with a different aggressively undersampled k-space trajectory. In this context, the single-coil acquisition model becomes:

.. math::

   \tilde{\boldsymbol{y}} = \begin{bmatrix}
      \mathcal{F}_{\Omega_1} & 0 & \cdots & 0 \\
      0 & \mathcal{F}_{\Omega_2} & \cdots & 0 \\
      \vdots & \vdots & \ddots & \vdots \\
      0 & 0 & \cdots & \mathcal{F}_{\Omega_T}
   \end{bmatrix}
   \boldsymbol{x} + \boldsymbol{n}

where :math:`\mathcal{F}_{\Omega_1}, \dots, \mathcal{F}_{\Omega_T}` are the Fourier operators corresponding to each individual frame. Some applications (e.g., MR Fingerprinting [3]_) may consists of 
thousands of total frames :math:`T`, leading to repeated Fourier Transform operations and high computational burden. However, the 1D signal series arising from similar voxels, e.g., with similar
relaxation properties, are typically highly correlated. For this reason, the image series can be represented as:

.. math::

   \boldsymbol{x} = \Phi\Phi^H \boldsymbol{x}

where :math:`\Phi` is an orthonormal basis spanning a low dimensional subspace whose rank :math:`K \ll T` which can be obtained performing a Singular Value Decomposition of a low resolution fully sampled
training dataset or an ensemble of simulated Bloch responses. The signal model can be then written as:

.. math::

   \tilde{\boldsymbol{y}} = \begin{bmatrix}
      \mathcal{F}_{\Omega_1} & 0 & \cdots & 0 \\
      0 & \mathcal{F}_{\Omega_2} & \cdots & 0 \\
      \vdots & \vdots & \ddots & \vdots \\
      0 & 0 & \cdots & \mathcal{F}_{\Omega_T}
   \end{bmatrix}
   \Phi \Phi^H \boldsymbol{x} + \boldsymbol{n} =
    \begin{bmatrix}
      \mathcal{F}_{\Omega_1} & 0 & \cdots & 0 \\
      0 & \mathcal{F}_{\Omega_2} & \cdots & 0 \\
      \vdots & \vdots & \ddots & \vdots \\
      0 & 0 & \cdots & \mathcal{F}_{\Omega_T}
   \end{bmatrix}
   \Phi \boldsymbol{\alpha} + \boldsymbol{n}

where :math:`\boldsymbol{\alpha} = \Phi^H \boldsymbol{x}` are the spatial coefficients representing the image series. Since the elements of :math:`\Phi^H` do not depend on the specific k-space frequency points,
the projection operator :math:`\boldsymbol{\Phi}` commutes with the Fourier transform, and the signal equation finally becomes:

.. math::

   \tilde{\boldsymbol{y}} = \Phi \mathcal{F}_\Omega(\boldsymbol{\alpha}) + \boldsymbol{n}

that is, computation now involves :math:`K \ll T` Fourier Transform operations, each with the same sampling trajectory, which can be computed by levaraging efficient NUFFT implementations for conventional static MRI.

The Non Uniform Fast Fourier Transform
======================================


In order to lower the computational cost of the Non-Uniform Fourier Transform, the main idea is to move back to a regular grid where an FFT would be performed (going from a typical :math:`O(MN)` complexity to :math:`O(M\log(N))`). Thus, the main steps of the *Non-Uniform Fast Fourier Transform* are for the type 1:

1. Spreading/Interpolation of the non-uniform points to an oversampled Cartesian grid (typically with twice the resolution of the final image)
2. Perform the (I)FFT on this image
3. Downsampling to the final grid, and apply some bias correction.

This package exposes interfaces to the main NUFFT libraries available. The choice of the spreading method (ie the interpolation kernel) in step 1. and the correction applied in step 3. are the main theoretical differences between the methods.

Type 2 transforms perform those steps in reversed order.

.. TODO Add Reference to all the NUFFT methods article
   Maybe to Fessler never-going-to-be-published book.


Density Compensation
====================




Other Application
=================
Apart from MRI, The NUFFT operator is also used for:

 - Electron tomography
 - Probability Density Function estimation
 - Astronomical Imaging
 - ...

These applications are not covered by this package, do it yourself!

References
==========

.. [1] https://en.m.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform
.. [2] Noll, D. C., Meyer, C. H., Pauly, J. M., Nishimura, D. G., Macovski, A., "A homogeneity correction method for magnetic resonance imaging with time-varying gradients", IEEE Transaction on Medical Imaging (1991), pp. 629-637.
.. [3] Fessler, J. A., Lee, S., Olafsson, V. T., Shi, H. R., Noll, D. C., "Toeplitz-based iterative image reconstruction for MRI with correction for magnetic field inhomogeneity",  IEEE Transactions on Signal Processing 53.9 (2005), pp. 3393–3402.
.. [4] D. F. McGivney et al., "SVD Compression for Magnetic Resonance Fingerprinting in the Time Domain," IEEE Transactions on Medical Imaging (2014), pp. 2311-2322.
