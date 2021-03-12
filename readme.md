# CBMOS

CBMOS is a Python framework for the numerical analysis of center-based models.
It focuses on flexibility and ease of use and is capable of simulating up to a
few thousand cells within a few seconds, or even up to 10,000 cells if GPU
support is available. CBMOS shines best for exploratory tasks and prototyping,
for instance when one wants to compare different sets of parameters or solvers.
At the moment, it implements most popular force functions, a few first and
second-order explicit solvers, and even one implicit solver. The following
sections describe how to run a simple simulation and illustrate what kind of
convergence studies can be performed with this package.

The package's documentation, as well as a few examples are available at
[somathias.github.io/cbmos/](https://somathias.github.io/cbmos/)

## Installation
### Install from source
From the root directory:
```
pip install .
```

Or, in developer mode:
```
pip install -e .
```

## Code structure

The *master* branch has the currently stable version of the code.

Basic examples illustrating how a basic model can be simulated and how to conduct a convergence study with CBMOS are available in the `example` folder.

## Unit Testing
Test are run through `pytest` or `python -m pytest`. Test functions are
automatically found by pytest (https://docs.pytest.org/en/latest/goodpractices.html#test-discovery). All the tests are run automatically on github upon pushing.

## Development 

Branches implementing new features should start with *dev-*. Currently we have 

 - *dev*: currently staging area for restructuring of the code into a python package
 - *dev-local_adaptivity*: started implementing local adaptivity

## Experiments

All experiments should be done in branches (off *master*, or *dev*) with names starting with *exp-*. We recommend running the experiments in jupyter notebooks for a nicer workflow :)

## Publications

- Mathias, S., Coulier, A., Bouchnita, A. et al. Impact of Force Function
  Formulations on the Numerical Simulation of Centre-Based Models. Bull Math
  Biol 82, 132 (2020). [DOI](https://doi.org/10.1007/s11538-020-00810-2) (tag `exp-Mathias2020`)
