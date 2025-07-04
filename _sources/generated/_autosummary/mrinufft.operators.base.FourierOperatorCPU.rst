FourierOperatorCPU
==================

.. currentmodule:: mrinufft.operators.base

.. autoclass:: FourierOperatorCPU
   :members:
   :private-members:
   :show-inheritance:
   :special-members: __call__, __add__, __mul__, __matmul__

   
   
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   
      ~FourierOperatorCPU.__init__
      ~FourierOperatorCPU.adj_op
      ~FourierOperatorCPU.cg
      ~FourierOperatorCPU.check_shape
      ~FourierOperatorCPU.compute_density
      ~FourierOperatorCPU.compute_smaps
      ~FourierOperatorCPU.data_consistency
      ~FourierOperatorCPU.get_lipschitz_cst
      ~FourierOperatorCPU.make_autograd
      ~FourierOperatorCPU.op
      ~FourierOperatorCPU.with_autograd
      ~FourierOperatorCPU.with_off_resonance_correction
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~FourierOperatorCPU.autograd_available
      ~FourierOperatorCPU.cpx_dtype
      ~FourierOperatorCPU.density
      ~FourierOperatorCPU.dtype
      ~FourierOperatorCPU.interfaces
      ~FourierOperatorCPU.n_coils
      ~FourierOperatorCPU.n_samples
      ~FourierOperatorCPU.ndim
      ~FourierOperatorCPU.norm_factor
      ~FourierOperatorCPU.samples
      ~FourierOperatorCPU.shape
      ~FourierOperatorCPU.smaps
      ~FourierOperatorCPU.uses_density
      ~FourierOperatorCPU.uses_sense
      ~FourierOperatorCPU.backend
      ~FourierOperatorCPU.available
   
   

   .. _sphx_glr_backref_mrinufft.operators.base.FourierOperatorCPU:

   .. minigallery:: mrinufft.operators.base.FourierOperatorCPU
      :add-heading: