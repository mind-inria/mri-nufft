.. include:: <isonum.txt>

===============================
MRI-NUFFT Interfaces Convention
===============================


The NUFFT Operator
==================


MRI-NUFFT provides a common interface for computing NUFFT, regardless of the chosen computation backend. All the backends implement the following methods and attributes. More technical details are available in the API reference.

All MRI-NUFFT operators inherit from :class:`FourierOperatorBase` . The minimum signature of an MRI nufft operator is:

.. code-block:: python

    class MRIOperator(FourierOperatorBase):

          backend = "my-nufft-backend"

          def __init__(samples: np.ndarray,
            shape:tuple[int,...],
            density: str | callable | np.ndarray | dict = False,
            n_coils: int = 1,
            n_batchs: int = 1,
            smaps: str | callable | np.ndarray | dict = False,
            ):

            self.samples = proper_trajectory(samples)
            self.shape = shape
            self.n_coils = n_coils
            self.n_batchs = n_batchs

            self.compute_density(density) # setup the density compensation
            self.compute_smaps(smaps) # setup the smaps

.. tip::

   The precision of the samples array will determine the precision of the computation. See also :ref:`K-Space Trajectories`

Moreover, the two following methods should be implemented for each backend

* ``op(image)`` : Forward Operation (image to k-space)
* ``adj_op(kspace)`` Adjoint Operation (k-space to image)

After initialization, defaults for the  following methods are available, as well as a range of QoL properties (``uses_sense``, ``uses_density``, ``ndim``, etc.).

* ``data_consistency(image, obs_data)``: perform the data consistency step  :math:`\cal{F}^*(\cal{F} x - y)`
* ``get_lipschitz_cst(max_iter)``: Estimate the spectral radius of the auto adjoint operator :math:`\cal{F}^*\cal{F}`

If the NUFFT backend makes some optimization possible, these backends can be manually overriden.



Extensions
----------

The base NUFFT operators can be extended to add extra functionality. With MRI-NUFFT we already provide:

- Off-resonnance Correction operators, using subspace separation (Sutton et al. IEEE TMI 2005)
- Auto-differentiation (for Deep Learning applications)


Adding a NUFFT Backend
----------------------

Adding a NUFFT backend to MRI-NUFFT should be easy. We recommend to check how other backends have been inplemented. CPU-based nufft interface can use the `FourierOperatorCPU` to minimize the boiler-plate.


K-Space Trajectories
====================

The k-space sampling trajectories are generated in the :py:mod:`mrinufft.trajectories` module and then used in the differents backends.
They are ``numpy`` arrays with the followings characteristics:

- ``float32`` or ``float64`` precision (this will trigger the use of single or double precision in the computations using this trajectory). ``float32`` precision is recommended for computational efficiency.

- They are row-major array (C Convention) with the following shape: ``(N_shot, N_samples, dim)`` (where  dim is either 2 or 3). A "flatten" version of shape ``(N_shots * N_samples, dim)`` is also acceptable by operators.

- By convention all k-space coordinates are in the range :math:`[-0.5,0.5)`. They will be rescaled internally by operator if required.
