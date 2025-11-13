Choosing a NUFFT Backend
========================

In order to perform Non-Uniform fast Fourier transform for MRI you need to install a computation library backend.


Supported Libraries
-------------------

These libraries need to be installed separately from this package.

.. Don't touch the spacing ! ..

==================== ============ =================== ===============  =================
Backend              Hardward     Batch computation   Precision        Array Interface
==================== ============ =================== ===============  =================
cufinufft_           GPU (CUDA)   ✔                   single           cupy/torch/numpy
finufft_             CPU          ✔                   single/double    numpy/torch
gpunufft_            GPU          ✔                   single/double    numpy/torch/cupy
tensorflow-nufft_    GPU (CUDA)   ✘                   single           tensorflow
pynufft-cpu_         CPU          ✘                   single/double    numpy
pynfft_              CPU          ✘                   single/double    numpy
bart_                CPU/GPU      ✔                   single           numpy
sigpy_               CPU          ✔                   single           numpy
stacked (*)          CPU/GPU      ✔                   single/double    numpy
==================== ============ =================== ===============  =================


.. _cufinufft: https://github.com/flatironinstitute/finufft
.. _finufft: https://github.com/flatironinstitute/finufft
.. _tensorflow-nufft: https://github.com/flatironinstitute/pynufft
.. _gpunufft: https://github.com/chaithyagr/gpuNUFFT
.. _pynufft-cpu: https://github.com/jyhmiinlin/pynufft
.. _pynfft: https://github.com/pynfft/pynfft
.. _bart: https://github.com/mrirecon/bart
.. _sigpy: https://github.com/sigpy/sigpy

- (*) stacked-nufft allows one to use any supported backend to perform a stack of 2D NUFFT and adds a z-axis FFT (using scipy or cupy)


**The NUFFT operation is often not enough to provide decent image quality by itself (even with density compensation)**.
For improved image quality, use a Compressed Sensing recon. For doing so, you can check the pysap-mri_ for MRI dedicated solutions and deepinv_ for Deep Learning based solutions.

.. _pysap-mri: https://github.com/CEA-COSMIC/pysap-mri/
.. _Modopt: https://github.com/CEA-COSMIC/ModOpt/
.. _deepinv: https:/github.com/deepinv/deepinv/

Backend Installations
---------------------

To benefit the most from certain backends we recommend to use the following instructions

finufft / cufinufft
~~~~~~~~~~~~~~~~~~~

Those are developed by the `flatiron-institute <https://github.com/flatironinstitute/finufft>`_ and are installable with `pip install finufft` and `pip install cufinufft`.

.. warning::

    for cufinufft, a working installation of CUDA and cupy is required.

gpuNUFFT
~~~~~~~~

an active gpuNUFFT fork is maintained by `chaithyagr <https://github.com/chaithyagr/gpunufft/>`_.


To install it use `pip install gpuNUFFT` or for local development, use the following: 

.. code-block:: sh

    git clone https://github.com/chaythiagr/gpuNUFFT
    cd gpuNUFFT
    python setup.py install

.. warning::

   If you are using ``uv`` as your package installer you will need to do ::

    .. code-block:: sh
    
         uv pip install wheel pip pybind11
         uv pip install mri-nufft[gpunufft] --no-build-isolation
    
BART
~~~~

BART has to be installed separately and `bart` command needs to be runnable from your `PATH`.
See `installation instructions <https://mrirecon.github.io/bart/installation.html>`_


PyNFFT
~~~~~~

PyNFFT requires Cython<3.0.0 to work.  and can be installed using

.. code-block:: sh

    pip install cython<3.0.0 pynfft2

Which backend to use
--------------------

We provided an extensive benchmark on computation and memory usage on https://github.com/mind-inria/mri-nufft-benchmark/

.. tip::

   Overall, we recommend to use ``finufft`` for CPU, and ``cufinufft`` or ``gpunufft`` when CUDA GPU are available.
