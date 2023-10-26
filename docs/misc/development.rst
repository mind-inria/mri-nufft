Developing MRI-nufft
====================

Developement environment
------------------------

We recommend to use a virtual environement, and install mri-nufft and its dependencies in it.


Running tests
-------------

Writing documentation
---------------------

Documentation is available online at https://mind-inria.github.io/mri-nufft

It can also be built locally ::

    cd mri-nufft
    pip install -e .[doc]
    python -m sphinx docs docs_build

To view the html doc locally you can use ::

    python -m http.server --directory docs_build 8000

And visit `localhost:8000` on your web browser.
