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

API stability and deprecations
------------------------------

MRI-nufft follows `effort-based versioning <https://docs.jax.dev/en/latest/jep/25516-effver.html>`_
rather than strict semantic versioning: a minor release may change the public API,
but such changes go through a deprecation period whenever possible.

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

This is enforced by:

- the *API check* CI job, which runs `griffe check <https://mkdocstrings.github.io/griffe/>`_
  on each pull request against the latest release, and fails on removed objects,
  moved parameters, changed defaults, etc. A breaking change that cannot go
  through a deprecation (or that griffe misreports, like an object kept through
  a module ``__getattr__``) is acknowledged by adding the line reported by the
  job to ``.github/api-breaks.txt``, with a comment saying why. Run it locally
  with ``.github/check_api.sh``.
- ``tests/test_deprecation.py``, which fails once a ``removed_in`` version is
  reached. It runs before each release is published.

Behaviour changes that keep the signatures untouched (a sign convention, an axis
order, ...) are invisible to these checks: flag them in the commit message with
``fix!:`` / ``feat!:`` or a ``BREAKING CHANGE:`` footer, so they end up in the
release notes.

Before a release ``X.Y.Z``, check the deprecation schedule and gather the breaking
changes for the release notes, from ``.github/api-breaks.txt`` and from the commits:

.. code-block:: bash

    MRINUFFT_RELEASE=X.Y.Z uv run pytest tests/test_deprecation.py
    git log --grep 'BREAKING CHANGE' --grep '^[a-z]+(\(.*\))?!:' -E $(git describe --tags --abbrev=0 --exclude '*-*')..HEAD

After the release, empty ``.github/api-breaks.txt`` (keeping its header): its
entries no longer apply against the new release, and the API check warns about
them.
