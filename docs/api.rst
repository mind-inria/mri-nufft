API Reference
=============

Reference API for MRI-NUFFT.

MRI-NUFFT public API is organized in modules [1]_, and the top level ``mrinufft`` only exposes :py:func:`~mrinufft.operators.get_operator`, which is the entry point to create NUFFT operator following the `NUFFT operator interface <mri-nufft-interface>`_

Here are the different modules that are the public API of MRI-NUFFT: 


.. autosummary::
   :toctree: generated/_autosummary

   mrinufft.operators
   mrinufft.trajectories
   mrinufft.display
   mrinufft.density
   mrinufft.io
   mrinufft.extras

.. [1] Like in `Scipy <https://docs.scipy.org/doc/scipy/reference/index.html#scipy-api>`_ 
