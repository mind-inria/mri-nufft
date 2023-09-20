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

These libraries needs to be installed separately from this package.

.. Don't touch the spacing ! ..

==================== ============ =================== =============== ============== ===============
Backend              Hardward     Batch computation   Precision       Auto Density   Array Interface
==================== ============ =================== =============== ============== ===============
cufinufft_           GPU (CUDA)   ✔                   single          ✔ *             cupy/torch
finufft_             CPU          ✔                   single/double   TBA            numpy
gpunufft_            GPU          ✔                   single/double   ✔              numpy
tensorflow-nufft_    GPU (CUDA)   ✘                   single          ✔              tensorflow
pynufft-cpu_         CPU          ✘                   single/double   ✘              numpy
pynfft_ (*)          CPU          ✘                   singles/double   ✘             numpy
stacked (**)         CPU/GPU      ✔                   single/double   ✔              numpy
==================== ============ =================== =============== ============== ===============


.. _cufinufft: https://github.com/flatironinstitute/finufft
.. _finufft: https://github.com/flatironinstitute/finufft
.. _tensorflow-nufft: https://github.com/flatironinstitute/pynufft
.. _gpunufft: https://github.com/chaithyagr/gpuNUFFT
.. _pynufft-cpu: https://github.com/jyhmiinlin/pynufft
.. _pynfft: https://github.com/ghisvail/pynfft

- (*) PyNFFT is only working with Cython < 3.0.0 , and is not actively maintained (https://github.com/mind-inria/mri-nufft/issues/19)
- (**) stacked-nufft allow to use any supported backend to perform a stack of 2D NUFFT and adds a z-axis FFT (using scipy)


**The NUFFT operation is often not enough to provide good image quality by itself (even with density compensation)**.  It is best used in a Compress Sensing setup, you can check the pysap-mri_ for MRI dedicated solutions and deepinv_ for Deep Learning based solutions.


Installation
------------
Install the required backend (e.g. `pip install finufft`) you want to use.

Then clone and install the package::

    git clone https://github.com:mind-inria/mri-nufft
    pip install ./mri-nufft

[Temporary] Custom Cufinufft installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


In order for the density compensation to work for cufinufft, we have to use the in-house fork enabling it ::

    git clone https://github.com/chaithyagr/finufft --branch chaithyagr/issue306
    cd finufft && mkdir build && cd build
    cmake -DFINUFFT_USE_CUDA=1 ../ && make -j && cp libcufinufft.so ../python/cufinufft/.
    cd ../python/cufinufft
    python setup.py install
    # Adapt to the name you have in python/cufinufft
    cp libcufinufft.so  cufinufftc.cpython-310-x86_64-linux-gnu.so

Development of this feature happens `here <https://github.com/flatironinstitute/finufft/pull/308>`_

[Temporary] Faster gpuNUFFT with concurency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A faster version of gpuNUFFT is available `here <https://github.com/chaithyagr/gpuNUFFT>`_.

.. warning::

    This is compatible only up to CUDA 11.8 !

To install it ::

    git clone https://github.com/chaythiagr/gpuNUFFT
    cd gpuNUFFT
    python setup.py install


Documentation
-------------

Documentation is available online at https://mind-inria.github.io/mri-nufft

It can also be built locally ::

    cd mri-nufft
    pip install -e .[doc]
    python -m sphinx docs docs_build

To view the html doc locally you can use ::

    python -m http.server --directory docs_build 8000

And visit `localhost:8000` on your web browser.


Related Packages
----------------

- pysap-mri_
- Modopt_
- deepinv_


.. _pysap-mri: https://github.com/CEA-COSMIC/pysap-mri/
.. _Modopt: https://github.com/CEA-COSMIC/ModOpt/
.. _deepinv: https:/github.com/deepinv/deepinv/
