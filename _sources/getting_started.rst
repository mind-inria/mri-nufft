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

In order to perform Non-Uniform fast Fourier transform you need to install a specific :ref:`NUFFT` computation library backend.

.. tip::

   TLDR: If you have a GPU and CUDA>=12.0, you probably want to install MRI-NUFFT like so:
   ``pip install mri-nufft[cufinufft]`` or ``pip install mri-nufft[gpunufft]``
   For CPU only setup we recommend ``pip install mri-nufft[finufft]``

   Then, use the ``get_operator(backend=<your backend>, ... )`` to initialize your MRI-NUFFT operator.

   For more information , check the :ref:`Examples`


Supported Libraries
-------------------

These libraries need to be installed separately from this package.

.. Don't touch the spacing ! ..

==================== ============ =================== ===============  =================
Backend              Hardward     Batch computation   Precision        Array Interface
==================== ============ =================== ===============  =================
cufinufft_           GPU (CUDA)   ✔                   single           cupy/torch/numpy
finufft_             CPU          ✔                   single/double    numpy
gpunufft_            GPU          ✔                   single/double    numpy
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

.. warning::

    This is compatible only up to CUDA 11.8 !

To install it use `pip install gpuNUFFT` or for local development.

.. code-block:: sh

    git clone https://github.com/chaythiagr/gpuNUFFT
    cd gpuNUFFT
    python setup.py install

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
