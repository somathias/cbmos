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

The *master* branch has the currently stable version of the code - with EF handling t_eval.

The *EF_for_benchmarks* branch has the currently stable version of the code - without EF handling t_eval, but with global adaptivity.

The main solver is found in cbmos_serial.py. Check it's __main__ function for an example on how to use it. All other file names aim to be self-explanatory. 

Files starting with *test_* are for testing the code in the correspoding python file using pytest (see below).

## Unit Testing
Test are run through `pytest` or `python -m pytest`. Test functions are
automatically found by pytest (https://docs.pytest.org/en/latest/goodpractices.html#test-discovery). All the tests are run automatically on bitbucket upon pushing.

## Development 

Branches implementing new features should start with *dev-*. Currently we have 

 - *dev*: currently staging area for restructuring of the code into a python package
 - *dev-local_adaptivity*: started implementing local adaptivity

## Experiments

All experiments should be done in branches (off *master*, *EF_for_benchmarks* or *dev*) with names starting with *exp-*. We recommend running the experiments in jupyter notebooks for a nicer workflow :)

