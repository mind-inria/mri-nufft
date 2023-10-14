Getting Started
===============

Installing MRI-NUFFT
--------------------

mri-nufft is available on PyPi

.. code-block:: sh

    pip install mri-nufft

Development Version
~~~~~~~~~~~~~~~~~~~

If you want to modifiy the mri-nufft code base

.. code-block:: sh

    git clone https://github.com:mind-inria/mri-nufft
    pip install -e ./mri-nufft[dev]


Choosing a NUFFT Backend
========================

In order to performs Non Uniform fast fourier transform you need to install a specific backend library.

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


**The NUFFT operation is often not enough to provide good image quality by itself (even with density compensation)**.
It is best used in a Compress Sensing setup, you can check the pysap-mri_ for MRI dedicated solutions and deepinv_ for Deep Learning based solutions.

.. _pysap-mri: https://github.com/CEA-COSMIC/pysap-mri/
.. _Modopt: https://github.com/CEA-COSMIC/ModOpt/
.. _deepinv: https:/github.com/deepinv/deepinv/

Custom Backend Installations
----------------------------

To benefit the most from certain backend we recommend to use the following instructions

[Temporary] Custom Cufinufft installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order for the density compensation to work for cufinufft, we have to use the in-house fork enabling it

.. code-block:: sh

    git clone https://github.com/chaithyagr/finufft --branch chaithyagr/issue306
    cd finufft && mkdir build && cd build
    cmake -DFINUFFT_USE_CUDA=1 ../ && make -j && cp libcufinufft.so ../python/cufinufft/.
    cd ../python/cufinufft
    python setup.py install
    # Adapt to the name you have in python/cufinufft
    cp libcufinufft.so  cufinufftc.cpython-310-x86_64-linux-gnu.so

Development of this feature happens on this `pull request <https://github.com/flatironinstitute/finufft/pull/308>`_

[Temporary] Faster gpuNUFFT with concurency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A faster version of gpuNUFFT is available on this `fork <https://github.com/chaithyagr/gpuNUFFT>`_.

.. warning::

    This is compatible only up to CUDA 11.8 !

To install it

.. code-block:: sh

    git clone https://github.com/chaythiagr/gpuNUFFT
    cd gpuNUFFT
    python setup.py install
