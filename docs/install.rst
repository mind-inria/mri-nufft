Installing MRI-NUFFT
====================

mri-nufft is available on `PyPi <https://pypi.org/project/mri-nufft/>`_


.. tip::

   TLDR: If you have a GPU and CUDA>=12.0, you probably want to install MRI-NUFFT with one of the two options: 

   - ``pip install mri-nufft[cufinufft]`` generally faster, but memory hungry
   - ``pip install mri-nufft[gpunufft]`` memory efficient, and still fast.
   - ``pip install mri-nufft[finufft]`` for non-cuda setup (CPU only) setup.

   Then, use the ``get_operator(backend=<your backend>, ... )`` to initialize your MRI-NUFFT operator.

   For more information , check the :ref:`general_examples`


.. code-block:: sh

    pip install mri-nufft

    
However, if you want to use some specific backends or develop on mri-nufft, you can install it with extra dependencies. notably `extra`, `io`,  and `autodiff`

.. code-block:: sh

    pip install mri-nufft[extra,io,autodiff]


Using ``uv``
~~~~~~~~~~~~
If you are using ``uv`` as your package installer you will need to do 
  
.. code-block:: sh

        uv pip install mri-nufft[extra,io,autodiff] --no-build-isolation

    
Development Version
~~~~~~~~~~~~~~~~~~~

If you want to modify the mri-nufft code base:

.. code-block:: sh

    git clone https://github.com:mind-inria/mri-nufft
    pip install -e ./mri-nufft[dev,doc,extra,io,autodiff,tests,cufinufft,gpunufft,finufft]

or using ``uv``: 

.. code-block:: sh

    git clone https://github.com:mind-inria/mri-nufft
    uv venv 
    uv sync --all-extras --no-build-isolation --no-extra <backend-you-dont-need>
