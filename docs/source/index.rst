.. CBMOS documentation master file, created by
   sphinx-quickstart on Thu Feb  4 11:28:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CBMOS' documentation!
=================================


CBMOS is a Python framework for the numerical analysis of center-based models.
It focuses on flexibility and ease of use and is capable of simulating up
to a few thousand cells within a few seconds, or even up to 10,000 cells if GPU support
is available. CBMOS shines best for exploratory tasks and prototyping, for
instance when one wants to compare different sets of parameters or solvers. At
the moment, it implements most popular force functions, a few first and second-order
explicit solvers, and even one implicit solver. The following sections describe
how to run a simple simulation and illustrate what kind of convergence studies
can be performed with this package.

.. include:: basic_example.rst

.. include:: convergence_example.rst

.. include:: generalized_events.rst

.. include:: modules.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
