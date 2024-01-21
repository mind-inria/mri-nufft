=========
MRI-NUFFT
=========

Doing non-Cartesian MR Imaging has never been so easy.

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - .. image:: https://mind-inria.github.io/mri-nufft/_static/coverage_badge.svg
     - .. image:: https://github.com/mind-inria/mri-nufft/workflows/CI/badge.svg
     - .. image:: https://github.com/mind-inria/mri-nufft/workflows/CD/badge.svg
   * - .. image:: https://img.shields.io/badge/style-black-black
     - .. image:: https://img.shields.io/badge/docs-Sphinx-blue
        :target: https://mind-inria.github.io/mri-nufft
     - .. image:: https://img.shields.io/pypi/v/mri-nufft
        :target: https://pypi.org/project/mri-nufft/


This python package extends various NUFFT (Non-Uniform Fast Fourier Transform) python bindings used for MRI reconstruction.

In particular, it provides a unified interface for all the methods, with extra features such as coil sensitivity, density compensated adjoint and off-resonance corrections (for static B0 inhomogeneities).


Usage
=====

.. TODO use a include file directive.
.. code-block:: python

      from scipy.datasets import face # For demo
      import numpy as np
      import mrinufft
      from mrinufft.trajectories import display
      from mrinufft.density import voronoi

      # Create a 2D Radial trajectory for demo
      samples_loc = mrinufft.initialize_2D_radial(Nc=100, Ns=500)
      # Get a 2D image for the demo (512x512)
      image = np.complex64(face(gray=True)[256:768, 256:768])

      ## The real deal starts here ##
      # Choose your NUFFT backend (installed independly from the package)
      NufftOperator = mrinufft.get_operator("finufft")

      # For better image quality we use a density compensation
      density = voronoi(samples_loc.reshape(-1, 2))

      # And create the associated operator.
      nufft = NufftOperator(
          samples_loc.reshape(-1, 2), shape=image.shape, density=density, n_coils=1
      )

      kspace_data = nufft.op(image)  # Image -> Kspace
      image2 = nufft.adj_op(kspace_data)  # Kspace -> Image


.. TODO Add image

For best image quality, embed these steps in a more complex reconstruction pipeline (for instance using `PySAP <https://github.com/CEA-COSMIC/pysap-mri>`_).

Want to see more ?

- Check the `Documentation <https://mind-inria.github.io/mri-nufft/>`_

- Or go visit the `Examples <https://mind-inria.github.io/mri-nufft/autoexamples/index.html>`_


Installation
------------

MRI-nufft is available on Pypi and can be installed with::

  pip install mri-nufft

You will also need to install at least one NUFFT computation backend. See the `Documentation <https://mind-inria.github.io/mri-nufft/getting_started.html#choosing-a-nufft-backend>`_ for more guidance.
