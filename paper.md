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
  - name: Asma Tanaben
    affiliation: 1 
  - name: Caini Pan 
    affiliation: 1 
  - name: Lena Oudjema 
    affiliation: 1
  - name: Mattheo Cencini
    affiliation: 4
  - name: Philippe Ciuciu
    affiliation: 1
  - name: Chaithya GR
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
    
affiliations:
 - name: MIND 
   index: 1
 - name: Université Paris-Saclay / CEA 
   index: 2
 - name: Chipiron
   index: 3
 - name: University Pisa 
   index: 4 
   
date: 20 September 2024
bibliography: paper.bib

---


# Summary 
MRI-NUFFT is a python package that provides a universal interface to various Non-Uniform Fast Fourier Transforms libraries running on CPU or GPU (gpunufft, finufft, cufinufft, pynfft), adding compabitily with standard array library (numpy, cupy, torch, tensorflow, etc.) On top of these librairies it extends the existing NUFFT operations to provides a physical model of the MRI acquisition processus (e.g. multi-coil acquisition and static-field inhomogeneities). Moreover it also provides a wide variety of non-Cartesian Sampling trajectories generation and expansion, as well as density-compensation estimation methods for those trajectories. It also implements optimized autodifferentiation with respect to the data and the sampling locations. With MRI-NUFFT one can experiment with non-cartesian sampling in MRI, get access to the latest advances in the field and state-of-the art sampling patterns.


# Statement of Need 
MRI is an non-invasive biomedical imaging technique, where raw data is sampled in the spatial frequency domain (k-space) and final images are  obtained by applying a (fast) fourier transform on this data.
Traditionnaly, the data is sampled on a cartesian grid, potentially with skipping lines (to accelerate the acquisition)  and reconstructed using FFT-based algorithms. 


# Main Characteristic of MRI-NUFFT 



# References



