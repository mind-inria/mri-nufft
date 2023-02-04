# MRI-NUFFT 

This python package extends various NUFFT (Non Uniform Fast Fourier Transform) python bindings for MRI Reconstruction usage. 

In particular it provides an unified interface for all the methods, with extra forward Model step, such as coil sensitivity, density compensated adjoint and Off Resonance corrections (B0 inhomogeneities)

Supported Library are: 
- GPU :
   - [cufinufft](https://github.com/flatironinstitute/cufinufft/) 
     Setup only single precision float32/complex64. Requires a separate installation of cufinufft C++/CUDA library. 
     
   - [tensorflow-nufft](https://github.com/mrphys/tensorflow-nufft) 
     Requires a separate installation of Tensorflow.
     
   - [TBA] [pyNUFFT](https://github.com/jyhmiinlin/pynufft) 
     (Not Yet Implemented)
   
     
- CPU : 
   - [finufft](https://github.com/flatironinstitute/finufft)
   - [TBA] [pyNUFFT](https://github.com/jyhmiinlin/pynufft)
     (Not Yet Implemented)

The NUFFT operation is often not enough to provide good image quality by itself. It is best used in an Compress Sensing setup. For such use cases,

you can check the [pysap](https://github.com/CEA-COSMIC/pysap/) package suite and  [pysap-mri](https://github.com/CEA-COSMIC/pysap-mri) for MRI dedicated solutions.

# Installation 

Be sure that you have your GPU librairies properly installed (CUDA, Pytorch, Tensorflow, etc).
Cufinufft requires an external installation.

Then clone and install the package
```shell
$ git clone https://github.com:paquiteau/mri-cufinufft
$ python mri-cufinufft/setup.py install 
```

# TODO 

- [x] Add support for finufft
- [ ] Expose a single Operator interface with `backend` argument and `**kwargs` for interface specific stuff.
- [ ] Add support for PyNUFFT CPU
- [ ] Add support for PyNUFFT GPU 
- [ ] Add density compensation estimation using standalone method (e.g. Voronoi)
- [ ] Add Documentation on NUFFT main principles (spread/ interpolation Kernel)
- [ ] Add benchmark 
