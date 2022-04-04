#!/usr/bin/env python

# System import
from __future__ import print_function
import os
from setuptools import setup, find_packages
try:
    from pip._internal.main import main as pip_main
except:
    from pip._internal import main as pip_main

# Global parameters
CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Environment :: Console",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering"]
AUTHOR = """
Pierre-Antoine Comby <pierre-antoine.comby@cea.fr>
"""
# Write setup
setup_requires = ["python-pysap"]

pip_main(['install'] + setup_requires)

setup(
    name="cufinufft-mri",
    description="Cufinufft extensions for MR Image processing",
    long_description="Cufinufft extensions for MR Imag processing",
    license="CeCILL-B",
    classifiers="CLASSIFIERS",
    author=AUTHOR,
    author_email="XXX",
    version="0.0.0",
    url="https://github.com/paquiteau/cufinufft-mri",
    packages=find_packages(),
    setup_requires=setup_requires,
    install_requires=[
    ],
    platforms="OS Independent"
)
