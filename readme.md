# CBMOS

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

Basic examples illustrating how a basic model can be simulated and how to conduct a convergence study with CBMOS are available in the `example` folder. A user guide will be available in the near future.

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
