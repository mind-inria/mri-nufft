# MRI-NUFFT 

This python package extends various NUFFT (Non Uniform Fast Fourier Transform) python bindings for MRI Reconstruction usage. 

In particular it provides an unified interface for all the methods, with extra forward Model step, such as coil sensitivity, density compensated adjoint and Off Resonance corrections (B0 inhomogeneities)

Supported Library are: 
- GPU :
   - [cufinufft](https://github.com/flatironinstitute/cufinufft/) 
     
     Only for  single precision float32/complex64. Requires a separate installation of cufinufft C++/CUDA library. 
     
   - [tensorflow-nufft](https://github.com/mrphys/tensorflow-nufft) 
     Requires a separate installation of Tensorflow.
     
   - [pyNUFFT](https://github.com/jyhmiinlin/pynufft) 
     (Not Yet Implemented)
   
   - [pygpuNUFFT](https://github.com/paquiteau/pygpuNUFFT)
     (Not Yet Implemented)
     
- CPU : 
   - [pyNUFFT](https://github.com/jyhmiinlin/pynufft)
   
     (Not Yet Implemented)

The NUFFT operation is often not enough to provide good image quality by itself. It is best used in an Compress Sensing setup. For such use cases,

you can check the [pysap](https://github.com/CEA-COSMIC/pysap/) package suite and  [pysap-mri](https://github.com/CEA-COSMIC/pysap-mri) for MRI dedicated solutions.

# Installation 

mriCufinufft has for dependencies:
- CUDA (Tested on 11.0)
- cupy 
- cufinufft 


Be sure that you have CUDA properly installed and install cufinufft in your virtual environment.

Then clone and install the package
```shell
$ git clone https://github.com:paquiteau/mri-cufinufft
$ python mri-cufinufft/setup.py install 
```

# Usage 

``` python
import cupy as cp
import numpy as np
from mriCufinufft import MRICufiNUFFT

shape = (512, 512) # img shape
sampling_ratio = 0.5
n_coils = 32

n_samples = int(np.prod(shape) * sampling_ratio)

smaps = np.random.randn(n_coils, *shape) + 1j * np.random.randn(n_coils, *shape)
smaps = smaps / np.linalg.norm(smaps, axis=0)
smaps = smaps.astype(np.complex64)

samples = np.random.uniform(-1, 1, (n_samples, len(shape))).astype(np.float32) * np.pi
fourier_op = MRICufiNUFFT(
    samples, shape,
    n_coils=n_coils,
    smaps=smaps,         # smaps are optional.
    smaps_cached=False,  # leave smaps on gpu.
    reuse_plans=True,    # reuse cufinufft internal plans.
    density=False,       # density compensated adjoint.
    eps=1e-5             # precision parameter.
)


image_data_sense = np.random.randn(*shape) + \
    1j * np.random.randn( *shape)
image_data_sense = np.squeeze(image_data).astype(np.complex64)

kspace_data = np.random.randn(n_coils, n_samples) + \
    1j * np.random.randn(n_coils, n_samples)
kspace_data = np.squeeze(kspace_data).astype(np.complex64)

adjoint = fourier_op.adj_op(kspace_data)

forward = fourier_op.op(image_data_sense)
```

See Also the `test` folder and the examples notebooks.


