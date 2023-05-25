=========
MRI-NUFFT
=========

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


This python package extends various NUFFT (Non Uniform Fast Fourier Transform) python bindings for MRI Reconstruction usage.

In particular it provides an unified interface for all the methods, with extra forward Model step, such as coil sensitivity, density compensated adjoint and Off Resonance corrections (B0 inhomogeneities)

Supported Library are:

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

The NUFFT operation is often not enough to provide good image quality by itself. It is best used in an Compress Sensing setup. For such use cases,

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


Documentation
-------------

Documentation is available online at https://mind-inria.github.io/mri-nufft

It can also be built locally ::

  $ cd mri-nufft
  $ pip install -e .[doc]
  $ python -m sphinx-build docs docs_build

To view the html doc locally you can use ::

  $ python -m http.server --directory docs_build 8000

And visit `localhost:8000` on your web browser.


Related Packages
----------------
For Reconstruction methods of MRI image from non cartesian sampling, see `pysap-mri <https://github.com/CEA-COSMIC/pysap-mri>`_ and `ModOpt <https://github.com/CEA-COSMIC/ModOpt>`_ 
