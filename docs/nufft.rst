The NUFFT Operator
==================

Definition
----------

The Non uniform Fast Fourier Transform (NUFFT) generalize the regular FFT operation where the point sampled in the Fourier domain are not uniformly distributed.

Application in MRI
------------------

In Magnetic Resonance Imaging (MRI) the raw data is acquired in the spatial Fourier domain (k-space).
Traditional sampling scheme of the k-space usually consist of acquired line in a specific  direction, in a Cartesian fashion.

In order to accelerate the acquisition of required data, using a non Cartesian (i.e. non uniformly distributed in *every* direction) sampling scheme offer great opportunity.

The acquisition model is usually described as:

.. math::

   y_i = \int_\mathbb{R}^d x(\bm{u}) e^{-2i\pi \bm{u} \cdot \bm{k_i}} d\bm{u} + n_i

Where:

- :math: `x(\bm{u})` is the spatially varying image contrast acquired.
- :math:`y_1, \dots, y_M` are the sampled points at locations :math:`\Omega=\lbrace k_1, \dots, k_n \in \mathbb{R}^d\rbrace`.
  Typically images (:math:`d=2`) or volumes (:math:`d=3`) are acquired.
- :math:`n_i` is a zero-mean complex valued Gaussian Noise, modeling the "thermal noise" of the scanner.

.. info::

   In order to reconstruct :math:`x` from :math:`y`, one has to solve the inverse problem, stated usually as:
   .. math::
      \hat{x} = \arg\min_x \frac12 \|\mathcal{F}_\Omega_\bm{x} - \bm{y}\|_2^2 + g(\bm{x}).

   This package focuses solely on computing :math:`\mathcal{F}\bm{x}` or :math:`\mathcal{F}^*\bm{y}`.
   solving this problem is **not** the purpose of the

Extension of the Acquisition model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The MRI Acquisition model can be extended in two main way. First by taking into account Parallel Imaging, where multiple coils are receiving data, each with a dedicated sensibility profile.

Parallel Imaging Model
""""""""""""""""""""""

Off-Resonance Correction Model
""""""""""""""""""""""""""""""



Numerical implementations of the NUFFT
--------------------------------------


Density Compensation
--------------------

Qualitative comparison
----------------------

See the future benchmark.




Other Application
-----------------
Apart from MRI, The NUFFT operator is also used for:

 - Electron tomography
 - Probability Density Function estimation
 - ...
These topic are not covered by this package, do it yourself !

References
----------
