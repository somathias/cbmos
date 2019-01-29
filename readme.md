# CBMOS

## Experiments

### Relaxation experiment (*exp-relaxation*)

Checkout *exp-relaxation* branch. Run **relaxation_experiment.py**.

### Force function plots (*exp-plot_forces*)

Plots the force laws for comparison. To do so, checkout the *exp-plot_forces* and run **plot_force_function.py**.
The first figure shows force laws fitted to cubic force law in both height and location of the maximum. 
The second figure shows force laws fitted only in height of the maximum and chosen to be small at the maximum interaction distance.
The third figure shows details of the general polynomial force law.

#### Parameter settings

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
    1. fit maximum in height and location: 
    2. fit maximum in height and make potential <10^-3 at rA=1.5:
  - quadratic:
