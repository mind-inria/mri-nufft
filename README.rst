
.. image:: https://github.com/mind-inria/mri-nufft/raw/master/docs/_static/logos/mri-nufft.png
    :width: 200px
    :align: center

|

|Docs| |PyPI| |JOSS| |Coverage| |CI| |CD| |Style| 


*Doing non-Cartesian MR Imaging has never been so easy*



Introduction
------------
   

MRI-NUFFT is an open-source Python library that provides state-of-the-art non-Cartesian MRI tools: trajectories, data loading and fast and memory-efficient operators to be used on laptops, clusters, and MRI consoles.

In particular, it provides a unified interface for computing Non-Uniform Fast Fourier Transform (`NUFFT <https://mind-inria.github.io/mri-nufft/nufft.html>`_), using the specialized backend of your choice (|finufft|_, |gpunufft|_, |torchkbnufft|_, ... ), and with integrated MRI-specific features such as:

- `multi-coil support <https://mind-inria.github.io/mri-nufft/generated/_autosummary/mrinufft.extras.smaps.html>`__ ,
- `density compensation <https://mind-inria.github.io/mri-nufft/generated/_autosummary/mrinufft.density.html>`__,
- `off-resonance correction <https://mind-inria.github.io/mri-nufft/generated/autoexamples/operators/example_offresonance.html>`__.
- `autodiff support <https://mind-inria.github.io/mri-nufft/generated/autoexamples/GPU/index.html>`__.

   
MRI-nufft is a nice and polite piece of software, that will return the same type of array (e.g ``numpy``, ``cupy``, ``torch``) provided at input, without extra copies for conversions.


On top of that we ship a variety of `non-Cartesian trajectories <https://mind-inria.github.io/mri-nufft/generated/autoexamples/trajectories/index.html>`__ commonly used by the MRI community, and even `tools <https://mind-inria.github.io/mri-nufft/generated/autoexamples/trajectories/example_trajectory_tools.html>`__ to helps you develop new ones. 

.. figure:: https://github.com/mind-inria/mri-nufft/raw/master/docs/_static/mri-nufft-scheme.png
   :width: 1000px
   :align: center
   
   *Modularity and Integration of MRI-nufft with the python computing libraries.*



Usage
-----

.. TODO use a include file directive.
.. code-block:: python

    from scipy.datasets import face # For demo
    import numpy as np
    import mrinufft

    # Create 2D Radial trajectories for demo
    samples_loc = mrinufft.initialize_2D_radial(Nc=100, Ns=500)
    # Get a 2D image for the demo (512x512)
    image = np.complex64(face(gray=True)[256:768, 256:768])

    ## The real deal starts here ##
    # Choose your NUFFT backend (installed independently from the package)
    # pip install mri-nufft[finufft] will be just fine here 
    nufft =  mrinufft.get_operator("finufft",
        samples_loc, shape=image.shape, density="voronoi", n_coils=1
    )

    kspace_data = nufft.op(image)  # Image -> Kspace
    image2 = nufft.adj_op(kspace_data)  # Kspace -> Image

    pinv = nufft.pinv_solver(kspace_data) # get a Pseudo inverse (least square minimization)

For improved image quality, embed these steps in a more complex reconstruction pipeline (for instance using `PySAP <https://github.com/CEA-COSMIC/pysap-mri>`_).

Want to see more ?

- Check the `Documentation <https://mind-inria.github.io/mri-nufft/>`__

- Or go visit the `Examples <https://mind-inria.github.io/mri-nufft/generated/autoexamples/index.html>`_


Installation
------------

MRI-nufft is available on `PyPi <https://pypi.org/project/mri-nufft>`__ and can be installed with::

  pip install mri-nufft

Additionally, you will have to install at least one NUFFT computation backend. See the `Documentation <https://mind-inria.github.io/mri-nufft/getting_started.html#choosing-a-nufft-backend>`__ for more guidance.
Typically we recommend:: 

  pip install mri-nufft[finufft] 
  pip install mri-nufft[cufinufft] # if you have a NVIDIA GPU and CUDA>=12


Benchmark
---------

A benchmark of NUFFT backend for MRI applications is available in https://github.com/mind-inria/mri-nufft-benchmark


Who is using MRI-NUFFT?
-----------------------

Here are several project that rely on MRI-NUFFT:

- `pysap-mri <https://github.com/CEA-COSMIC/pysap-mri>`_
- `snake-fmri <https://github.com/paquiteau/snake-fmri>`_
- `deepinv <https://github.com/deepinv/deepinv>`_

Add yours by opening a PR or an issue,  let us know how you use MRI-nufft ! 


How to cite MRI-NUFFT
---------------------

We published MRI-NUFFT at `JOSS <https://doi.org/10.21105/joss.07743>`__ :: 

    Comby et al., (2025). MRI-NUFFT: Doing non-Cartesian MRI has never been easier. Journal of Open Source Software, 10(108), 7743, https://doi.org/10.21105/joss.07743

.. code:: bibtex
          
    @article{Comby2025, doi = {10.21105/joss.07743},
        author = {Comby, Pierre-Antoine and Daval-Frérot, Guillaume and Pan, Caini and Tanabene, Asma and Oudjman, Léna and Cencini, Matteo and Ciuciu, Philippe and GR, Chaithya},
        title = {MRI-NUFFT: Doing non-Cartesian MRI has never been easier}, journal = {Journal of Open Source Software},
        url = {https://doi.org/10.21105/joss.07743},
        year = {2025},
        publisher = {The Open Journal},
        volume = {10},
        number = {108},
        pages = {7743},
    } 


Contributing
------------

We warmly welcome contributions ! Check out our `guidelines <https://github.com/mind-inria/mri-nufft/blob/master/CONTRIBUTING.md>`_ , 
Don't hesitate to look for unsolved `issues <https://github.com/mind-inria/mri-nufft/issues/>`__




.. |Coverage| image:: https://raw.githubusercontent.com/mind-inria/mri-nufft/refs/heads/colab-examples/examples/_static/coverage_badge.svg
.. |CI| image:: https://github.com/mind-inria/mri-nufft/actions/workflows/test-ci.yml/badge.svg
.. |CD| image:: https://github.com/mind-inria/mri-nufft/workflows/CD/badge.svg
.. |Style| image:: https://img.shields.io/badge/style-black-black
.. |Docs| image:: https://img.shields.io/badge/docs-Sphinx-blue
    :target: https://mind-inria.github.io/mri-nufft
.. |PyPI| image:: https://img.shields.io/pypi/v/mri-nufft
    :target: https://pypi.org/project/mri-nufft/
.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.07743/status.svg
    :target: https://doi.org/10.21105/joss.07743

.. |finufft| replace:: ``(cu)finufft``
.. _finufft: https://github.com/flatironinstitute/finufft

.. |gpunufft| replace:: ``gpunufft``
.. _gpunufft: https://github.com/chaithyagr/gpunufft

.. |torchkbnufft| replace:: ``torchkbnufft``
.. _torchkbnufft: https://github.com/mmuckley/torchkbnufft


