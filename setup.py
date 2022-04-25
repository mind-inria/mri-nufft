#!/usr/bin/env python

import os
import sys
import io

from setuptools import find_packages, setup, Command

# Package meta-data
NAME = "mriCufinufft"
DESCRIPTION = "Non Uniform Fourier Transform for MRI. Based on cufinufft."
URL = "https://github.com/paquiteau/mriCufinufft"
EMAIL = "pierre-antoine.comby@cea.fr"
AUTHOR = "Pierre-Antoine Comby"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.0.0"
CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Environment :: Console",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering"]

# Required packages
REQUIRED = [
    "numpy",
    "cupy",
    "python-pysap",
]
# Nice to have
EXTRAS = [
    "pysap-mri",
    "pysap-fmri",
]


here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require={'all':EXTRAS},
    include_package_data=True,
)
