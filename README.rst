=========
MRI-NUFFT
=========

Doing non-Cartesian MR Imaging has never been so easy.

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - .. image:: https://img.shields.io/badge/coverage-TBA-green
        :target: https://app.codecov.io/gh/mind-inria/mri-nufft
     - .. image:: https://github.com/mind-inria/mri-nufft/workflows/CI/badge.svg
     - .. image:: https://github.com/mind-inria/mri-nufft/workflows/CD/badge.svg
   * - .. image:: https://img.shields.io/badge/style-black-black
     - .. image:: https://img.shields.io/badge/docs-Sphinx-blue
        :target: https://mind-inria.github.io/mri-nufft
     - .. image:: https://img.shields.io/pypi/v/mri-nufft
        :target: https://pypi.org/project/mri-nufft/


This python package extends various NUFFT (Non-Uniform Fast Fourier Transform) python bindings used for MRI reconstruction.

In particular, it provides a unified interface for all the methods, with extra features such as coil sensitivity, density compensated adjoint and off-resonance corrections (for B0 inhomogeneities).


Usage
=====

.. TODO use a include file directive.
.. code:: python

      from scipy.datasets import face # For demo
      import numpy as np
      import mrinufft
      from mrinufft.trajectories import display
      from mrinufft.trajectories.density import voronoi

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

- Or go visit the `Examples <https://mind-inria.github.io/mri-nufft/auto_examples/index.html>`_

Supported Libraries
-------------------

These libraries needs to be installed seperately from this package.

- GPU Implementations:

  - `cufinufft <https://github.com/flatironinstitute/cufinufft/>`_
      Developed and maintained by the `Flat Iron Institute <https://github.com/flatironinstitut>`_.
      Requires a separate installation of cufinufft C++/CUDA library.
      Current bindings only support float32/complex64 data.

  - `tensorflow-nufft <https://github.com/mrphys/tensorflow-nufft>`_
      Requires a separate installation of Tensorflow.

  - TBA `pyNUFFT <https://github.com/jyhmiinlin/pynufft>`_
      (Not Yet Implemented)

- CPU Implementations:

  - `finufft <https://github.com/flatironinstitute/finufft>`_
      Developed and maintained by the `Flat Iron Institute <https://github.com/flatironinstitut>`_.
      C/C++ implementation with Multithread and batch computation support.

  - `pyNUFFT <https://github.com/jyhmiinlin/pynufft>`_
      CPU version of pyNUFFT, using standard python libraries.

The NUFFT operation is often not enough to provide good image quality by itself: It is best used in a Compress Sensing setup. For such use cases,

you can check the `pysap <https://github.com/CEA-COSMIC/pysap/>`_ package suite and  `pysap-mri <https://github.com/CEA-COSMIC/pysap-mri>`_ for MRI dedicated solutions.

Installation
------------

Be sure that you have your GPU librairies properly installed (CUDA, Pytorch, Tensorflow, etc).
Cufinufft requires an external installation.

Then clone and install the package::

    $ git clone https://github.com:mind-inria/mri-nufft
    $ pip install ./mri-nufft

Tests
-----
TBA


Documentation
-------------

Documentation is available online at https://mind-inria.github.io/mri-nufft

It can also be built locally ::

  $ cd mri-nufft
  $ pip install -e .[doc]
  $ python -m sphinx docs docs_build

To view the html doc locally you can use ::

  $ python -m http.server --directory docs_build 8000

And visit `localhost:8000` on your web browser.


Related Packages
----------------
For reconstruction methods of MR images from non-Cartesian sampling, see `pysap-mri <https://github.com/CEA-COSMIC/pysap-mri>`_ and `ModOpt <https://github.com/CEA-COSMIC/ModOpt>`_ 
