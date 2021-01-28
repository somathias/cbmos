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

 - *dev-local_adaptivity*: started implementing local adaptivity
 - *force_functions*: started work on implementing the derivatives needed for implicit solvers

## Experiments

All experiments should be done in branches (off *master*) with names starting with *exp-*. We recommend running the experiments in jupyter notebooks for a nicer workflow :)

### Relaxation experiment (*exp-relaxation*)

Checkout *exp-relaxation* branch. Open the jupyter notebook *exp-relaxation.ipynb*. 

**Note** that this branch does not yet use the new code implementing proliferation (everything after commit 7197dd7).

### Adhesion experiment (*exp-adhesion*)
Checkout *exp-adhesion* branch. Open the jupyter notebook *exp-adhesion.ipynb*. 

**Note** that this branch does not yet use the new code implementing proliferation (everything after commit 7197dd7).

### Collapsing volumes (*exp-collapsing_volumes*)
Checkout *exp-collapsing_volumes* branch. There are three jupyter notebooks. Note that *exp-collapsing_volumes_1d.ipynb* and *exp-collapsing_volumes_2d.ipynb* need to be updated to work with the newer code (which has been merged into this branch). 
*exp-collapsing_volumes_sheet.ipynb* is up-to-date and should run. 

### Tumor growth (*exp-tumor_growth*)
Checkout *exp-tumor_growth* branch. Open the jupyter notebook *exp-tumor_growth.ipynb*.

Note the changes in cell.py. Cells have a default mean cell cycle duration of 6 hours in this branch (normal distribution N(6, 0.25)). 

### Force function plots (*exp-plot_forces*)

Plots the force laws for comparison. To do so, checkout the *exp-plot_forces* and run **plot_force_function.py**.
The first figure shows force laws fitted to cubic force law in both height and location of the maximum. 
The second figure shows force laws fitted only in height of the maximum and chosen to be small at the maximum interaction distance.
The third figure shows details of the general polynomial force law.

#### Parameter settings (for *exp-plot_forces*)

  We fix 

  - s = 1.0  # equilibrium rest length, set to 1 cell diameter
  - rA = s+0.5  # set maximum interaction distance to 1.5 cell diameter
  - rR = s+0.2  # set maximum repulsive interaction distance to 1.2 cell diameter
  - rN = 0.3  # radius of nuclei for hard-core model

  Then we use the cubic force law as a basis, since it only has a single free parameter left (the spring stiffness mu).

  We can fix the force amplitude such that f_max^cubic = 1.0, ie then everything is in relation to the maximum force value of the cubic force law.

  The maximum of the cubic force law is attained at r=7/6 for the above parameter values. Its value is mu_cubic/54, hence 

  f_max^cubic = 1.0 <=> mu_cubic=54

  Now we fit all other force laws to this.

  - lennard-jones: m = 1.0
  - morse: two options
    1. fit maximum in height and location: m =1.0, a = 6*(log2)
    2. fit maximum in height and make potential <10^-3 at rA=1.5:
       m=1.0, a = 16.58759909 (found by scipy.optimize.minimize with
       BFGS method)
  - linear-exponential: same two options
    1. fit maximum in height and location: mu = 6*e, a = 6
    2. fit maximum in height and make potential <10^-3 at rA=1.5:
       mu=55.63460379 (found by scipy.optimize.minimize with BFGS
       method), a=-2*np.log(0.002/mu)
  - piecewise quadratic: fit maximum in height: muR = 84, muA= 0.25*mR = 21 (location of maximum is then r=8/7)


