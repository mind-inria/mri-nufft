Developing MRI-nufft
====================

If you are a first time contributor, we recommend you to check the `CONTRIBUTING.md <https://github.com/mind-inria/mri-nufft/blob/master/CONTRIBUTING.md>`_ file.

Developement environment
------------------------

We recommend to use a virtual environement, and install mri-nufft and its dependencies in it.

Running tests
-------------

To run tests you will need to install the `test` extra dependencies with e.g. `pip install -e .[test]` 

You can run the tests with `pytest`. Don't hesitate to check how our `CI <https://github.com/mind-inria/mri-nufft/blob/master/.github/workflows/test-ci.yml>`_ runs tests for more details. 

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
