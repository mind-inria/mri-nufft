MRIGpuNUFFT
===========

.. currentmodule:: mrinufft.operators.interfaces.gpunufft

.. autoclass:: MRIGpuNUFFT
   :members:
   :private-members:
   :show-inheritance:
   :special-members: __call__, __add__, __mul__, __matmul__

   
   
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   
      ~MRIGpuNUFFT.__init__
      ~MRIGpuNUFFT.adj_op
      ~MRIGpuNUFFT.cg
      ~MRIGpuNUFFT.check_shape
      ~MRIGpuNUFFT.compute_density
      ~MRIGpuNUFFT.compute_smaps
      ~MRIGpuNUFFT.data_consistency
      ~MRIGpuNUFFT.get_lipschitz_cst
      ~MRIGpuNUFFT.make_autograd
      ~MRIGpuNUFFT.op
      ~MRIGpuNUFFT.pipe
      ~MRIGpuNUFFT.toggle_grad_traj
      ~MRIGpuNUFFT.with_autograd
      ~MRIGpuNUFFT.with_off_resonance_correction
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~MRIGpuNUFFT.autograd_available
      ~MRIGpuNUFFT.available
      ~MRIGpuNUFFT.backend
      ~MRIGpuNUFFT.cpx_dtype
      ~MRIGpuNUFFT.density
      ~MRIGpuNUFFT.dtype
      ~MRIGpuNUFFT.interfaces
      ~MRIGpuNUFFT.n_batchs
      ~MRIGpuNUFFT.n_coils
      ~MRIGpuNUFFT.n_samples
      ~MRIGpuNUFFT.ndim
      ~MRIGpuNUFFT.norm_factor
      ~MRIGpuNUFFT.samples
      ~MRIGpuNUFFT.shape
      ~MRIGpuNUFFT.smaps
      ~MRIGpuNUFFT.uses_density
      ~MRIGpuNUFFT.uses_sense
   
   

   .. _sphx_glr_backref_mrinufft.operators.interfaces.gpunufft.MRIGpuNUFFT:

   .. minigallery:: mrinufft.operators.interfaces.gpunufft.MRIGpuNUFFT
      :add-heading: