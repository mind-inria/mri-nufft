#---
title: 'MRI-NUFFT: Doing Non Cartesian MRI has never been easier'
tags:
  - Python
  - MRI
  - NUFFT
  - Numpy 
  - CUDA
  - Torch 
authors:
  - name: Pierre-Antoine Comby
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Guillaume Daval-Frérot
    affiliation: 3
  - name: Chaithya GR
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Philippe Ciuciu
    affiliation: 1
    
affiliations:
 - name: MIND 
   index: 1
 - name: Université Paris-Saclay / CEA 
   index: 2
 - name: Chipiron
   index: 3

date: 25 December 2023
bibliography: paper.bib

---


# Summary 
MRI is an non-invasive biomedical imaging technique, where raw data is sampled in the spatial frequency domain (k-space) and final images are  obtained by applying a (fast) fourier transform on this data. Traditionnaly, the 

MRI-NUFFT is a python package that provides a universal interface to various Non-Uniform Fast Fourier Transforms libraries running on CPU or GPU, adding compabitily with standard array library (numpy, cupy, torch, tensorflow, pyCUDA and numba. On top of these librarires it extends the existing NUFFT operations to provides a physical model of the MRI acquisition processus (e.g. multi coil acquisition and static-field inhomogeneities). Moreover it also provides a wide variety of non-Cartesian Sampling trajectories generation and expansion, as well as density-compensation estimation methods for those trajectories.


# Statement of Need 


# Main Characteristic of MRI-NUFFT 



# References



