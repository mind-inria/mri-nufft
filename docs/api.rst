API Reference
=============

Reference API for MRI-NUFFT.

MRI-NUFFT public API is organized in modules [1]_, and the top level ``mrinufft`` only exposes a handfull of utilities functions, that had nowhere else to go. 

.. tip::
   
   From the top level you likely only need :py:func:`~mrinufft.operators.get_operator`,
   the entry point to create NUFFT operator following the `NUFFT operator interface <mri-nufft-interface>`_

Here are the different modules that are the public API of MRI-NUFFT: 

.. autosummary::
   :toctree: generated/_autosummary


.. autosummary::
   :toctree: generated/_autosummary
             
   mrinufft 
   mrinufft.operators
   mrinufft.trajectories
   mrinufft.display
   mrinufft.density
   mrinufft.io
   mrinufft.extras

.. [1] Like in `Scipy <https://docs.scipy.org/doc/scipy/reference/index.html#scipy-api>`_ 
