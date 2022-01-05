# NPL

NPL (NanoParticleLibrary) is a wrapper around the popular ASE library to offer a convenient way to work with nanoparticles. It was developed specifically
for the optimization of the chemical ordering in bimetallic nanoparticles. For this purpose it includes: 
  - a flexible energy pipeline for approximate energy models
  - Local optimization of the ordering of nanoparticles
  - pre-built optimization algorithms such as Markov Chain Monte Carlo and Genetic Algorithms
  - Basin-Hopping utilizing the local optimization procedure for both relaxation and perturbation
  - Built-in procedure for shape optimization of a fixed lattice structure (such as FCC, icosahedral, decahedral, ...)
  - Memory efficient storage of large training sets
  
# Installation
NPL requires the Atomic Simulation Environment >= 3.21 and Python 3.7+ as well as scikit-learn and the sortedcontainers library by Grant Jenks.

# Overview
In the following the most important modules and their functionality will be briefly described. Demonstrations in the form of Jupyter notebooks can be found in the
tutorials folder. 

## BaseNanoparticle
  

 
