Installing MRI-NUFFT
====================

mri-nufft is available on `PyPi <https://pypi.org/project/mri-nufft/>`_


.. tip::

   TLDR: If you have a GPU you probably want to install MRI-NUFFT with one of the following options:
   
   - ``uv add mri-nufft[cufinufft]`` generally faster, but memory hungry
   - ``uv add mri-nufft[gpunufft]`` memory efficient, and still fast.
   - ``uv add mri-nufft[finufft]`` for non-cuda setup (CPU only) setup.

   (the regular `pip install` or `uv pip install` would work as well).
     
   Then, use the ``get_operator(backend=<your backend>, ... )`` to initialize your MRI-NUFFT operator.

   You are ready to use MRI-NUFFT ! Go check the :ref:`general_examples` for more information.


.. code-block:: sh

  pip install mri-nufft

    
However, if you want to use some specific backend or develop on mri-nufft, you can install it with extra dependencies. notably `extra`, `io`,  and `autodiff`

.. code-block:: sh

   uv add mri-nufft[extra,io,autodiff]

.. tip::

  using uv will ensure that the correction version of pytorch (with CUDA) is installed. for more info see `this < https://docs.astral.sh/uv/guides/integration/pytorch/>`__
  

Using ``uv``
~~~~~~~~~~~~
If you are using ``uv`` as your package installer you will need to do 
  
.. code-block:: sh

        uv add mri-nufft[extra,io,autodiff]

    
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


Contributing to MRI-NUFFT
~~~~~~~~~~~~~~~~~~~~~~~~~

Check our `CONTRIBUTING.md <https://github.com/mind-inria/mri-nufft/blob/master/CONTRIBUTING.md>`__
