# NanoParticleLibrary (NPL)

NPL (NanoParticleLibrary) is a comprehensive wrapper around the popular ASE (Atomic Simulation Environment) library, designed to facilitate the manipulation and optimization of nanoparticles. It is particularly tailored for optimizing the chemical ordering in bimetallic nanoparticles. 

## Features

NPL offers a variety of features to support nanoparticle research and optimization:

- **Flexible Energy Pipeline**: Supports approximate energy models for efficient computations.
- **Local Optimization**: Optimizes the ordering of particles locally.
- **Pre-built Optimization Algorithms**: Includes algorithms such as Markov Chain Monte Carlo (MCMC) and Genetic Algorithms (GA).
- **Basin-Hopping**: Utilizes local optimization for both relaxation and perturbation steps.
- **Shape Optimization**: Provides built-in procedures for optimizing the shape of fixed lattice structures (e.g., FCC, icosahedral, decahedral).
- **Memory Efficient Storage**: Efficiently stores large training sets to save memory.

## Installation

To install NPL, ensure you have the following dependencies:

- Python 3.7+
- Atomic Simulation Environment (ASE) >= 3.21
- scikit-learn
- sortedcontainers (by Grant Jenks)

You can install the required packages using pip:

```bash
pip install ase>=3.21 scikit-learn sortedcontainers

Overview
Below is a brief overview of the most important modules and their functionality:

Core Modules: Provide the fundamental classes and functions for nanoparticle manipulation.
Optimization Algorithms: Implement various optimization techniques such as Markov Chain Monte Carlo (MCMC) and Genetic Algorithms (GA).
Energy Models: Offer different energy computation models for simulations.
Shape Optimization: Tools for optimizing the shape of particles.
Storage Utilities: Efficient storage mechanisms for large datasets.
Usage
To use NPL, import the necessary modules and start by creating and optimizing a nanoparticle. Here is a simple example:

from npl.core import Nanoparticle
from npl.optimization import MonteCarlo

# Create a nanoparticle instance
nanoparticle = Nanoparticle(...)

# Perform optimization using Monte Carlo
mc_optimizer = MonteCarlo(nanoparticle)
mc_optimizer.optimize()


Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure your code follows the project's coding standards and includes appropriate tests.

License
NPL is licensed under the MIT License. See the LICENSE file for more details.

Contact
For any questions or issues, please open an issue on GitHub or contact the maintainers.