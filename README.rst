=========
MRI-NUFFT
=========

Doing non-Cartesian MR Imaging has never been so easy.

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - .. image:: https://mind-inria.github.io/mri-nufft/_static/coverage_badge.svg
     - .. image:: https://github.com/mind-inria/mri-nufft/actions/workflows/test-ci.yml/badge.svg
     - .. image:: https://github.com/mind-inria/mri-nufft/workflows/CD/badge.svg
   * - .. image:: https://img.shields.io/badge/style-black-black
     - .. image:: https://img.shields.io/badge/docs-Sphinx-blue
        :target: https://mind-inria.github.io/mri-nufft
     - .. image:: https://img.shields.io/pypi/v/mri-nufft
        :target: https://pypi.org/project/mri-nufft/


This python package extends various NUFFT (Non-Uniform Fast Fourier Transform) python bindings used for MRI reconstruction.

In particular, it provides a unified interface for all the methods, with extra features such as coil sensitivity, density compensated adjoint and off-resonance corrections (for static B0 inhomogeneities).

.. raw:: html 
   
   <div align="center">

.. image:: https://github.com/mind-inria/mri-nufft/raw/master/docs/_static/mri-nufft-scheme.svg
   :width: 700
   :align: center

Modularity and Integration of MRI-nufft with the python computing libraries.

.. raw:: html 
   
   </div>

Usage
=====

.. TODO use a include file directive.
.. code-block:: python

      from scipy.datasets import face # For demo
      import numpy as np
      import mrinufft
      from mrinufft.trajectories import display
      from mrinufft.density import voronoi

      # Create 2D Radial trajectories for demo
      samples_loc = mrinufft.initialize_2D_radial(Nc=100, Ns=500)
      # Get a 2D image for the demo (512x512)
      image = np.complex64(face(gray=True)[256:768, 256:768])

      ## The real deal starts here ##
      # Choose your NUFFT backend (installed independently from the package)
      NufftOperator = mrinufft.get_operator("finufft")

      # For improved image reconstruction, use density compensation
      density = voronoi(samples_loc.reshape(-1, 2))

      # And create the associated operator.
      nufft = NufftOperator(
          samples_loc.reshape(-1, 2), shape=image.shape, density=density, n_coils=1
      )

      kspace_data = nufft.op(image)  # Image -> Kspace
      image2 = nufft.adj_op(kspace_data)  # Kspace -> Image


.. TODO Add image

For improved image quality, embed these steps in a more complex reconstruction pipeline (for instance using `PySAP <https://github.com/CEA-COSMIC/pysap-mri>`_).

Want to see more ?

- Check the `Documentation <https://mind-inria.github.io/mri-nufft/>`_

- Or go visit the `Examples <https://mind-inria.github.io/mri-nufft/generated/autoexamples/index.html>`_


Installation
------------

MRI-nufft is available on Pypi and can be installed with::

  pip install mri-nufft

Additionally, you will have to install at least one NUFFT computation backend. See the `Documentation <https://mind-inria.github.io/mri-nufft/getting_started.html#choosing-a-nufft-backend>`_ for more guidance.


Benchmark
---------

A benchmark of NUFFT backend for MRI applications is available in https://github.com/mind-inria/mri-nufft-benchmark


Who is using MRI-NUFFT?
-----------------------

Here are several project that rely on MRI-NUFFT:

- `pysap-mri <https://github.com/CEA-COSMIC/pysap-mri>`_
- `snake-fmri <https://github.com/paquiteau/snake-fmri>`_
- `deepinv <https://github.com/deepinv/deepinv>`_


  Add yours by opening a PR !
