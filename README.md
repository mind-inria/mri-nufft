# MRI Cufinufft

This is a python  package which extend the [cufinufft](https://github.com/flatironinstitute/cufinufft/) to be efficiently use for MRI reconstruction.


For high level utilization you can check the [pysap](https://github.com/CEA-COSMIC/pysap/) package suite, in particular the `NonCartesianFFT` class in [pysap-mri](https://github.com/CEA-COSMIC/pysap-mri).

# Installation 

First install cufinufft (be sure to compile it for you GPU architecture), and its python bindings.

Then clone and install the package
``` shell
$ git clone github.com/paquiteau/mri-cufinufft
$ python mri-cufinufft/setup.py install 
```


