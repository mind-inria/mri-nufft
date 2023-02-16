The NUFFT Operator
==================

This document gives a general overview of the Non Uniform Fast fourier transform (NUFFT), and its application in MRI.


The Non Uniform Discrete Fourier Transform
------------------------------------------

The non uniform discrete fourier transform (NUDFT) is a generalization of the discrete fourier transform (DFT) to non uniform sampling. The NUDFT is defined as:



https://en.m.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform

Application in MRI
------------------

In Magnetic Resonance Imaging (MRI) the raw data is acquired in the spatial Fourier domain (k-space).
Traditional sampling scheme of the k-space usually consist of acquired line in a specific  direction, in a Cartesian fashion.

In order to accelerate the acquisition of required data, using a non Cartesian (i.e. non uniformly distributed in *every* direction) sampling scheme offer great opportunity.

The acquisition model is usually described as:

.. math::
   y(\boldsymbol{\nu}_i) = \int_{\mathbb{R}^d} x(\boldsymbol{u}) e^{-2i\pi \boldsymbol{u} \cdot \boldsymbol{\nu_i}} d\boldsymbol{u} + n_i, \quad i=1,\dots,M

Where:

- :math:`x(\boldsymbol{u})` is the spatially varying image contrast acquired.
- :math:`y_1, \dots, y_M` are the sampled points at frequency locations :math:`\Omega=\lbrace \boldsymbol{\nu}_1, \dots, \boldsymbol{\nu}_M \in [-1/2, 1/2]^d\rbrace`.
  Typically images (:math:`d=2`) or volumes (:math:`d=3`) are acquired.
- :math:`n_i` is a zero-mean complex valued Gaussian Noise, modeling the "thermal noise" of the scanner.


In practice the equation above is discretized, and the integral is replaced by a sum, using a finite number of samples :math:`N`:

.. math::
    y(\boldsymbol{\nu}_i) = \sum_{j=1}^N x(\boldsymbol{u}_j) e^{-2i\pi\boldsymbol{u}_j\cdot\boldsymbol{\nu_i}} + n_i, \quad i=1,\dots,M

Where :math:`\boldsymbol{u}_j` are the :math:`N` spatial locations of image voxels.
This is stated using the operator notation:

.. math::
    \boldsymbol{y} = \mathcal{F}_\Omega (\boldsymbol{x}) + \boldsymbol{n}

As the sampling locations :math:`\Omega` are non uniform and the image locations :math:`\boldsymbol{u}_j` are uniform, :math:`\mathcal{F}_\Omega` is a NUDFT operator, and the equation above describe a type II NUDFT.


.. attention::

   In order to reconstruct :math:`x` from :math:`y`, one has to solve the *inverse problem*, stated usually as

   .. math::
      \hat{x} = \arg\min_x \frac{1}{2} \|\mathcal{F}_\Omega(\boldsymbol{x}) - \boldsymbol{y}\|_2^2 + g(\boldsymbol{x}).

   This package focuses solely on computing :math:`\mathcal{F}_\Omega\boldsymbol{x}` or :math:`\mathcal{F}_\Omega^**\boldsymbol{y}`.
   solving this problem is **not** addressed here, but you can check `pysap-mri <https://github.com/CEA-COSMIC/pysap-mri>`_ for this purpose.

Extension of the Acquisition model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The MRI Acquisition model can be extended in two main way. First by taking into account Parallel Imaging, where multiple coils are receiving data, each with a dedicated sensibility profile.

Parallel Imaging Model
""""""""""""""""""""""

In MRI the acquired signal can be received by multiple antenna ("coils"). Each coils possess a specific sensitivity profile (each sees the object differently due to their physical layout).

The Acquisition model for parallel imaging with :math:`L` coils  is:

.. math::

   y_{i,\ell} = \int_{\mathbb{R}^d} S_\ell(\boldsymbol{u})x(\boldsymbol{u}) e^{-2i\pi \boldsymbol{u} \cdot \boldsymbol{k_i}} d\boldsymbol{u} + n_{i,\ell}

Or using the operator notation:

.. math::

   \tilde{\boldsymbol{y}} = \begin{bmatrix}
    \mathcal{F}_\Omega S_1 \\
    \vdots  \\
    \mathcal{F}_\Omega S_L \\
    \end{bmatrix}
    \boldsymbol{x} + \boldsymbol{n}_\ell  = \mathcal{F}_\Omega S \otimes \boldsymbol{x} + \tilde{\boldsymbol{n}}

Where :math:`S_1, \dots, S_L` are the sensitivity maps of each coils. Such sensitivity maps can be acquired separetely by acquiring low frequency of the kspace, or estimated from the data.

..
    TODO Add ref to SENSE and CG-Sense

Off-Resonance Correction Model
""""""""""""""""""""""""""""""

..
    See ref in Guillaume Daval-Frerot Thesis


The general NUFFT algorithm
---------------------------


Density Compensation
--------------------




Other Application
-----------------
Apart from MRI, The NUFFT operator is also used for:

 - Electron tomography
 - Probability Density Function estimation
 - ...

These applications are not covered by this package, do it yourself !

References
----------
