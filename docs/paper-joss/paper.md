---
title: 'MRI-NUFFT: Doing non-Cartesian MRI has never been easier'
tags:
  - Python
  - MRI
  - NUFFT
  - Numpy 
  - CUDA
  - Torch 
authors:
  - name: Pierre-Antoine Comby
    orcid: 0000-0001-6998-232X
    corresponding: true
    affiliation: "1, 2"
  - name: Guillaume Daval-Frérot
    affiliation: 3
  - name: Asma Tanaben
    affiliation: "1,2,5" 
  - name: Caini Pan 
    affiliation: "1, 2"
  - name: Lena Oudjema 
    affiliation: 1,2
  - name: Matteo Cencini
    affiliation: 4
  - name: Philippe Ciuciu
    affiliation: "1,2"
  - name: Chaithya GR
    corresponding: true 
    affiliation: "1,2"
    
affiliations:
 - name: MIND, Inria
   index: 1
 - name: Université Paris-Saclay / CEA 
   index: 2
 - name: Chipiron
   index: 3
 - name: INFN, Pisa Division 
   index: 4 
 - name: Siemens Healthineers 
   index: 5
   
date: 20 September 2024
bibliography: paper.bib
---


# Summary 
MRI-NUFFT is a python package that provides a universal interface to various Non-Uniform Fast Fourier Transforms libraries running on CPU or GPU (gpunufft, finufft, cufinufft, pynfft), adding compatibily with standard array library (numpy, cupy, torch, tensorflow, etc.) On top of these libraries it extends the existing NUFFT operations to provide a physical model of the MRI acquisition process (e.g. multi-coil acquisition and static-field inhomogeneities). Moreover, it also provides a wide variety of non-Cartesian Sampling trajectories generation and expansion, as well as density-compensation estimation methods for those trajectories. It also implements optimized auto-differentiation with respect to the data and the sampling locations. With MRI-NUFFT one can experiment with non-cartesian sampling in MRI, get access to the latest advances in the field and state-of-the art sampling patterns.


# Statement of Need 
MRI is an non-invasive biomedical imaging technique, where raw data is sampled in the spatial frequency domain (k-space) and final images are  obtained by applying a (fast) fourier transform on this data.
Traditionnaly, the data is sampled on a Cartesian grid, potentially with skipping lines (to accelerate the acquisition)  and reconstructed using FFT-based algorithms. 
However, the Cartesian grid is not always the best choice for sampling the data, and non-cartesian sampling schemes have been proposed to improve the image quality, reduce the acquisition time or to enable new imaging modalities. The reconstruction of non-cartesian data is more complex than the cartesian case, and requires the use of non-uniform fast Fourier transform (NUFFT) algorithms. 
Several NUFFT libraries have been developed in the past years, but they are not always easy to use or implementing the specificity of MRI data acquisition (e.g. multi-coil acquisition, static-field inhomogeneities, density compensation, etc.). Also their performances can vary a lot depending on the specific use-cases (2D vs 3D data, number of coils, etc.). 

Moreover, the use of non-cartesian sampling in MRI is still an active research field, with new sampling patterns being proposed regularly. With MRI-NUFFT one can  easily experiment with these new patterns, and compare them with existing one.
Furthermore, there has been a growing interest in using deep learning for MRI acquisition and reconstruction, and using those new methods for Non-Cartesian Data requires to be able to compute the gradients of the reconstruction with respect to the data and/or the sampling locations.

# Features 

![MRI-NUFFT as an interface for Non Cartesian MRI](../_static/mri-nufft-scheme.svg)

## NUFFT Library compatibility 
MRI-NUFFT is compatible with the following NUFFT librairies: finufft[@barnett_parallel_2019], cufinufft[@shih_cufinufft_2021], gpunufft[@knoll_gpunufft_2014], torchkbnufft[@muckley_torchkbnufft_2020], pynfft, sigpy[@ong_frank_sigpy_2019] and BART[@uecker_berkley_2015]. 
Using our [benchmark](https://github.com/mind-inria/mri-nufft-benchmark/) we can also determined which implementations of the NUFFT provides the best performances (both in term of computation time and memory footprint). As the time of writing cufinufft and gpunufft provides the best performances, by leveraging CUDA acceleration. MRI-NUFFT supports as well  standard array libraries (numpy, cupy, torch, tensorflow, etc.) and optimizes data copies, relying on the array-api standard. 
On top of these NUFFT backend, it provides several enhancements, notably an optimized 2.5D NUFFT (for stack of 2D non uniform trajectory, a common pattern in MRI), and a data-consistency term for iterative reeconstruction ($\mathcal{F}_\Omega^*(\mathcal{F}_\Omega x - y)$) that can be used in iterative reconstruction algorithms.


## Extended Fourier Model 
MRI-NUFFT provides a physical model of the MRI acquisition processus, including multi-coil acquisition and static-field inhomogeneities. This model is compatible with the NUFFT libraries, and can be used to simulate the acquisition of MRI data, or to reconstruct data from a given set of measurements. Namely we provide a linear operator that encapsulates the forward and adjoint NUFFT operators, the coil sensitivity maps and (optionnaly) the static field inhomogeneities. The forward model is described by the following equation:
$$y(\boldsymbol{\nu}_i) = \sum_{j=1}^N x(\boldsymbol{u}_j) e^{-2\imath\pi\boldsymbol{u}_j\cdot\boldsymbol{\nu_i}} + n_i, \quad i=1,\dots,M$$
Where:
$x(\boldsymbol{u})$ is the spatially varying image contrast acquired; $y_1, \dots, y_M$ are the sampled points at frequency locations;  $\Omega=\lbrace \boldsymbol{\nu}_1, \dots, \boldsymbol{\nu}_M \in [-1/2, 1/2]^d\rbrace$; $\boldsymbol{u}_j$ are the $N$ spatial locations of image voxels.; and $n_i$ is a zero-mean complex-valued Gaussian Noise, modeling the thermal noise of the scanner.

This can also be formulated using the operator notation $\boldsymbol{y} = \mathcal{F}_\Omega (\boldsymbol{x}) + \boldsymbol{n}$

As the sampling locations $\Omega$ are non-uniform and the image locations $\boldsymbol{u}_j$ are uniform, $\mathcal{F}_\Omega$ is a NUDFT operator, and the equation above describe a Type 2 NUDFT.
Similarly the adjoint operator is a Type 1 NUFFT:

: Correspondance Table between NUFFT and MRI acquisition model.

| NUFFT Type | Operation | MRI Transform      | Operator               |
|:-----------|:----------|:-------------------|:-----------------------|
| Type 1     | Adjoint   | Kspace $\to$ Image | $\mathcal{F}_\Omega^*$ |
| Type 2     | Forward   | Image $\to$ Kspace | $\mathcal{F}_\Omega$   |


### Parallel Imaging Model
In MRI the acquired signal can be received by multiple antennas (\"coils\"). 
Each coil possesses a specific sensitivity profile (i.e. each sees the object differently due to its physical layout).

$$\begin{aligned}
\tilde{\boldsymbol{y}} = \begin{bmatrix}
 \mathcal{F}_\Omega S_1 \\
 \vdots  \\
 \mathcal{F}_\Omega S_L \\
 \end{bmatrix}
 \boldsymbol{x} + \boldsymbol{n}_\ell  = \mathcal{F}_\Omega S \otimes \boldsymbol{x} + \tilde{\boldsymbol{n}}
\end{aligned}$$

Where $S_1, \dots, S_L$ are the sensitivity maps of each coil. 
Such sensitivity maps can be acquired separetely by acquiring low frequency of the kspace, or estimated from the data.

### Off-resonance correction model
The constant magnetic field applied in a MRI machine $B0$ (with a typical intensity 1.5, 3 or 7 Tesla) is inherently disturbed at tissue interfaces with owing to different magnetic susceptibilities (such as air-tissue interfaces in the nose and ear canals). 
Those field perturbations introduce a spatially varying phase shift in the frequencies acquired (noted $\Delta\omega_0$), making the acquisition model deviating from the convenient Fourier model. 
Fortunately, this inhomogeneity map can be acquired separately or estimated and integrated in the model as:

$$y(t_i) = \int_{\mathbb{R}^d} x(\boldsymbol{u}) e^{-2\imath\pi \boldsymbol{u} \cdot\boldsymbol{\nu_i} + \Delta\omega(\boldsymbol{u}) t_i} d\boldsymbol{u}$$

where $t_i$ is the time at which the frequency $\nu_i$ is acquired.
With these mixed-domain field pertubations, the Fourier model does not hold anymore and the FFT algorithm cannot be used any longer. 
The main approach [@sutton_fast_2003] is to approximate the mixed-domain exponential term by splitting it into single-domain weights $b_{m, \ell}$ and $c_{\ell, n}, where $L \ll M, N$ regular Fourier transforms are performed to approximate the non-Fourier transform.

$$x(\boldsymbol{u_n}) = \sum_{\ell=1}^L c_{\ell, n} \sum_{m}^M y(t_m) b_{m, \ell} e^{2\imath\pi \boldsymbol{u} \cdot \boldsymbol{\nu_i}}$$

The coefficients $B=(b_{m, \ell}) \in \mathbb{C}^{M\times L}$ and $C=(c_\ell, n) \in \mathbb{C}^{L\times N}$ can be optimally estimated within MRI-NUFFT.

## Trajectories generation and expansions 
MRI-NUFFT comes with a wide variety of Non Cartesian trajectory generation routines, that have been gathered from the literature. It also provides ways of expanding existing trajectories. It is also able to export to specific formats, to be used in other toolboxes and on MRI hardware.

## Autodifferentiation for data and sampling pattern

Following the formulation of [@wang_efficient_2023], MRI-NUFFT also provides autodifferentation capabilities for all the NUFFT backends. Both gradients with respect to the data (image or kspace) and the sampling point location are available. This allows for efficient backpropagation throught the NUFFT operators, and sustains research on learned sampling pattern and image reconstruction network.

# MRI-NUFFT Utilization
MRI-NUFFT is already used in conjunction with other software such as SNAKE-fMRI [@comby_snake-fmri_2024], deepinv [@tachella_deepinverse_2023] and PySAP-MRI [@farrens_pysap_2020]

# References



