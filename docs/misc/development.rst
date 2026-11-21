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

.. _api-stability:

API stability and deprecations
------------------------------

MRI-nufft follows `effort-based versioning <https://docs.jax.dev/en/latest/jep/25516-effver.html>`_
rather than strict semantic versioning: a minor release may change the public API,
but such changes go through a deprecation period whenever possible, and are always
listed in the :doc:`changelog <../changelog>`.

The public API is everything exported in the ``__all__`` of a public (not
underscored) module. A deprecated part of it keeps working, and emits a
``mrinufft._deprecation.MRINufftDeprecationWarning`` (a ``FutureWarning``, shown
by default), for **at least two minor releases** before its removal: deprecated in
``1.7.0``, removed in ``1.9.0`` at the earliest.

Use the helpers of ``mrinufft._deprecation``, which keep the messages consistent:

.. code-block:: python

    from mrinufft._deprecation import deprecated, moved_attribute, renamed_parameter

    @deprecated(since="1.7.0", removed_in="1.9.0", alternative="`new_func`")
    def old_func(...): ...

    @renamed_parameter("n", "n_bins", since="1.7.0", removed_in="1.9.0")
    def func(..., n_bins=-1): ...

    # In the module the object moved away from:
    def __getattr__(name):
        if name == "OldName":
            return moved_attribute(
                __name__, name, "mrinufft.new.module", since="1.7.0", removed_in="1.9.0"
            )
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

New parameters should be added after the existing ones, preferably as keyword-only,
so that positional calls keep their meaning.

When a pull request breaks or deprecates part of the public API, including
behaviour changes that leave the signatures untouched (a new default, a sign
convention, an axis order...), add an entry to the *Unreleased* section of
``docs/changelog.rst``.

Making a release
----------------

The version number comes from the git tag, through ``setuptools_scm``: there is
nothing to bump.

1. Prepare the release in a pull request:

   .. code-block:: bash

       git fetch origin
       git switch -c release-X.Y.Z origin/master
       uv run --extra test tools/release.py prepare X.Y.Z
       git commit -am "chore: release X.Y.Z"
       gh pr create --fill

   ``prepare`` stops if a deprecated object is due for removal in ``X.Y.Z``: remove
   it first. It then renames the *Unreleased* section of ``docs/changelog.rst`` to
   ``X.Y.Z (date)``, and opens a new *Unreleased* section. Check that this section
   is complete before merging.

2. Tag the merged commit and push the tag:

   .. code-block:: bash

       git switch master && git pull
       git tag -a vX.Y.Z -m "vX.Y.Z"
       git push origin vX.Y.Z

3. The ``tags-release.yml`` workflow drafts the GitHub release, with the changelog
   section followed by the list of pull requests and new contributors, then
   publishes the package on PyPI. It publishes nothing if the changelog has no
   ``X.Y.Z`` section.

4. Proofread and publish the draft release, which also archives it on Zenodo.
