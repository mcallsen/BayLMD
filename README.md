# Overview

BayLMD implements dynamic training of machine-learning based force fields using Bayesian linear regression[^1].

The force field is based on the usual lattice dynamics expansion of the total energy in terms of displacements of
the atoms, where the parameters of the model are the components of the force constant tensors. The dynamic training 
then searches the configuration space for optimal training data using Bayesian error estimation.

# Installation

The package requires `mpi4py` for parallelization, `ASE` for the MD and running ab initio calculations, `csld` 
for creating the force fields, and `pybind11` for linking the python and the C++ library.

> [!NOTE]
> For performance reasons we recommend avoiding `conda` for installing certain packages. In particular, the `mpi4py` 
> package installed using `conda` uses a basic, precompiled MPI library. Instead, install `mpi4py` using `pip` and 
> the MPI compiler on your cluster.

Compile the C++ library using
```
make lib
```
in the `source` directory. The MPI Compiler and library locations can be specified in `makefile.include`. Finally, 
install the python package using 
```
pip install baylmd
```
[^1]: M. Callsen, T. T. Lee, M. Y. Chou, [arXiv:2601.21301 (2026)](https://arxiv.org/abs/2601.21311)
