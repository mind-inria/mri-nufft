FourierOperatorBase
===================

.. currentmodule:: mrinufft.operators.base

.. autoclass:: FourierOperatorBase
   :members:
   :private-members:
   :show-inheritance:
   :special-members: __call__, __add__, __mul__, __matmul__

   
   
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   
      ~FourierOperatorBase.__init__
      ~FourierOperatorBase.adj_op
      ~FourierOperatorBase.cg
      ~FourierOperatorBase.check_shape
      ~FourierOperatorBase.compute_density
      ~FourierOperatorBase.compute_smaps
      ~FourierOperatorBase.data_consistency
      ~FourierOperatorBase.get_lipschitz_cst
      ~FourierOperatorBase.make_autograd
      ~FourierOperatorBase.op
      ~FourierOperatorBase.with_autograd
      ~FourierOperatorBase.with_off_resonance_correction
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~FourierOperatorBase.autograd_available
      ~FourierOperatorBase.cpx_dtype
      ~FourierOperatorBase.density
      ~FourierOperatorBase.dtype
      ~FourierOperatorBase.interfaces
      ~FourierOperatorBase.n_batchs
      ~FourierOperatorBase.n_coils
      ~FourierOperatorBase.n_samples
      ~FourierOperatorBase.ndim
      ~FourierOperatorBase.norm_factor
      ~FourierOperatorBase.samples
      ~FourierOperatorBase.shape
      ~FourierOperatorBase.smaps
      ~FourierOperatorBase.uses_density
      ~FourierOperatorBase.uses_sense
      ~FourierOperatorBase.backend
      ~FourierOperatorBase.available
   
   

   .. _sphx_glr_backref_mrinufft.operators.base.FourierOperatorBase:

   .. minigallery:: mrinufft.operators.base.FourierOperatorBase
      :add-heading: