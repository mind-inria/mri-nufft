---
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
    affiliation: "1,2,5" 
  - name: Caini Pan 
    affiliation: "1, 2"
  - name: Lena Oudjema 
    affiliation: 1,2
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
 - name: Siemens Healthineers 
   index: 5
   
date: 20 September 2024
bibliography: paper.bib
---


# Summary 
MRI-NUFFT is a python package that provides a universal interface to various Non-Uniform Fast Fourier Transforms libraries running on CPU or GPU (gpunufft, finufft, cufinufft, pynfft), adding compabitily with standard array library (numpy, cupy, torch, tensorflow, etc.) On top of these librairies it extends the existing NUFFT operations to provides a physical model of the MRI acquisition processus (e.g. multi-coil acquisition and static-field inhomogeneities). Moreover it also provides a wide variety of non-Cartesian Sampling trajectories generation and expansion, as well as density-compensation estimation methods for those trajectories. It also implements optimized autodifferentiation with respect to the data and the sampling locations. With MRI-NUFFT one can experiment with non-cartesian sampling in MRI, get access to the latest advances in the field and state-of-the art sampling patterns.


# Statement of Need 
MRI is an non-invasive biomedical imaging technique, where raw data is sampled in the spatial frequency domain (k-space) and final images are  obtained by applying a (fast) fourier transform on this data.
Traditionnaly, the data is sampled on a cartesian grid, potentially with skipping lines (to accelerate the acquisition)  and reconstructed using FFT-based algorithms. 
However, the cartesian grid is not always the best choice for sampling the data, and non-cartesian sampling schemes have been proposed to improve the image quality, reduce the acquisition time or to enable new imaging modalities. The reconstruction of non-cartesian data is more complex than the cartesian case, and requires the use of non-uniform fast fourier transform (NUFFT) algorithms. 
Several NUFFT libraries have been developed in the past years, but they are not always easy to use, and they are not always implementing the specificities of MRI data acquisition (e.g. multi-coil acquisition, static-field inhomogeneities, density compensation, etc.). Also their performances can vary a lot depending on the specific use-case (2D vs 3D data, number of coils, etc.). 

Moreover, the use of non-cartesian sampling in MRI is still an active research field, with new sampling patterns being proposed regularly. It is important for researchers to be able to easily experiment with these new patterns, and to compare them with existing ones. Recently there has been a growing interest in using deep learning for MRI acquisition and reconstruction, and using those new methods for Non-Cartesian Data requires to be able to compute the gradients of the reconstruction with respect to the data and/or the sampling locations. Some attemps have been made, but their implementation remains either slow, wrong or lack documentation. 



# Features 


## Extended Fourier Model 



## NUFFT libraries compatibility

## Trajectories generation and expansions 

## Density compensation estimation

## Autodifferentiation


# References



