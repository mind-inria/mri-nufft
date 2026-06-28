
.. image:: https://github.com/mind-inria/mri-nufft/raw/master/docs/_static/logos/mri-nufft.png
    :width: 200px
    :align: center

|

|Docs| |PyPI|  |colab|

|CI| |CD| |Style| |Coverage| 

|JOSS| |OSI|

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
Recommended: uv
~~~~~~~~~~~~~~~

You can install MRI-nufft in a virtual environment using `uv <https://astral-sh/uv>`__ with::

  uv venv # create a virtual env if needed
  uv add mri-nufft[finufft,cufinufft,extra,autodiff]


Regular pip install
~~~~~~~~~~~~~~~~~~~
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
.. |PyPI| image:: https://img.shields.io/pypi/dm/mri-nufft.svg?logo=pypi&label=pip%20install&color=fedcba
    :target: https://pypi.org/project/mri-nufft/
.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.07743/status.svg
    :target: https://doi.org/10.21105/joss.07743
.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/mind-inria/mri-nufft/blob/colab-examples/examples/generated/autoexamples/operators/example_readme.ipynb

.. |OSI| image:: https://img.shields.io/badge/DIN%20SPEC%203105-OSI²_CAB:_1.1/60-3498DB?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4bWw6c3BhY2U9InByZXNlcnZlIiBpZD0iRWJlbmVfMSIgeD0iMCIgeT0iMCIgc3R5bGU9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgNzcwLjEgNzcwIiB2ZXJzaW9uPSIxLjEiIHZpZXdCb3g9IjAgMCA3NzAuMSA3NzAiPjxzdHlsZT4uc3Qxe2NsaXAtcGF0aDp1cmwoI1NWR0lEXzJfKTtmaWxsOiMzNzUzNmN9PC9zdHlsZT48ZGVmcz48cGF0aCBpZD0iU1ZHSURfMV8iIGQ9Ik0wIDBoNzcwdjc3MEgweiIvPjwvZGVmcz48Y2xpcFBhdGggaWQ9IlNWR0lEXzJfIj48dXNlIHhsaW5rOmhyZWY9IiNTVkdJRF8xXyIgc3R5bGU9Im92ZXJmbG93OnZpc2libGUiLz48L2NsaXBQYXRoPjxwYXRoIGQ9Ik00MTEgMjk2YTEwMiAxMDIgMCAwIDEgMzkgMjIgOTIgOTIgMCAwIDEgMTMgMTE3bC04IDExYy00IDUtOSA5LTE0IDEybDE4IDIyIDExIDE0IDQtM2ExMzcgMTM3IDAgMCAwIDQ5LTEwNmMwLTU1LTM1LTExOS0xMTAtMTM2bC0xOS0yaC0xOGMtMTAgMS0yMCAyLTMwIDUtMTggNS0zNyAxNC01NCAzMGExMzQgMTM0IDAgMCAwLTQ1IDkzdjIwbDIgMTkgMyA5YTEyOSAxMjkgMCAwIDAgNDcgNzBsMTAtMTMgMTgtMjNhOTIgOTIgMCAwIDEtMzItNDdsLTEtNC0yLTE2di0xMGwyLTE0YTkyIDkyIDAgMCAxIDExNy03MCIgc3R5bGU9ImNsaXAtcGF0aDp1cmwoI1NWR0lEXzJfKTtmaWxsOiMzYTk3ZDMiLz48cGF0aCBkPSJNNDI0IDcxNWEzMzAgMzMwIDAgMCAwIDI5MS0yODlsLTQ1LTZhMjgyIDI4MiAwIDAgMS0xNjYgMjI2Yy0yOCAxMy01NyAyMS04NiAyNGw2IDQ1ek01NSA0MjZhMzMwIDMzMCAwIDAgMCAyODggMjg5bDYtNDZhMjg5IDI4OSAwIDAgMS0yNDgtMjQ3bC0xLTItNDUgNnpNMzQ0IDU1QTMzNSAzMzUgMCAwIDAgNTUgMzQ2bDQ2IDVhMjgyIDI4MiAwIDAgMSA5OC0xODVjNDktNDEgMTAwLTU5IDE0OS02NmgxbC01LTQ1em03NSA0NWEyODIgMjgyIDAgMCAxIDI0MiAyMDcgMjQ4IDI0OCAwIDAgMSA5IDQ0bDQ1LTV2LTFjLTYtNTctMzgtMTQwLTk1LTE5NS02Ni02My0xMzEtODctMTk2LTk1bC02IDQ1aDF6IiBjbGFzcz0ic3QxIi8+PHBhdGggZD0iTTU2MCAzMDRhMTk0IDE5NCAwIDAgMS0yMyAyMDFsMzUgMjcgNi04YTIzNCAyMzQgMCAwIDAgNDUtMTU2IDIzMyAyMzMgMCAwIDAtNTEtMTMxbC0zNiAyOWM5IDEyIDIwIDI5IDI0IDM4TTUzMSAxOTdjLTE1LTExLTMwLTIxLTQ4LTI5bC0yMC04LTIwLTZhMjQyIDI0MiAwIDAgMC0yMDUgNDRsMjggMzZhMTkwIDE5MCAwIDAgMSAyMzctMWwyOC0zNnpNMTk4IDIzOWE0ODQgNDg0IDAgMCAwLTI0IDM2bC0xMCAyMmEyMzYgMjM2IDAgMCAwIDMzIDIzM2wzNi0yOC0xNC0yMGMtMjEtMzQtMzUtOTMtMjEtMTQ0bDYtMTggOC0yMGEyOTcgMjk3IDAgMCAxIDIyLTMzbC0zNi0yOHoiIGNsYXNzPSJzdDEiLz48cGF0aCBkPSJNNTMxIDU3M2MtMTUgMTEtMzAgMjAtNDggMjhsLTIwIDktMjAgNWEyNDIgMjQyIDAgMCAxLTIwNS00NGwyOC0zNmExOTAgMTkwIDAgMCAwIDIzNyAybDI4IDM2eiIgc3R5bGU9ImNsaXAtcGF0aDp1cmwoI1NWR0lEXzJfKTtmaWxsOiMzODk1ZDIiLz48L3N2Zz4=&labelColor=white&link=https://gitlab.com/osiiev/cab/-/blob/main/review/projects/MRI-NUFFT/v1.5.0/review_report.md
   :target: https://www.opensourceimaging.org/project/mri-nufft



.. |finufft| replace:: ``(cu)finufft``
.. _finufft: https://github.com/flatironinstitute/finufft

.. |gpunufft| replace:: ``gpunufft``
.. _gpunufft: https://github.com/chaithyagr/gpunufft

.. |torchkbnufft| replace:: ``torchkbnufft``
.. _torchkbnufft: https://github.com/mmuckley/torchkbnufft


