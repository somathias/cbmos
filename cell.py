#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:00:48 2019

@author: Sonja Mathias

The average cell cycle duration for a typical rapidly proliferating human cell
is 24 hours (https://bionumbers.hms.harvard.edu/bionumber.aspx?id=112260).

We assume a normal distribution of cell cycle durations with mean=24h and
sigma=1.0h.
"""

import numpy as np


class Cell:
    """
    Parameters
    ----------
    position : numpy array
        The position of this cell (1, 2, or 3d)
    division_time : float
        Absolute simulation time when division will take place for this cell


    """
    def __init__(self, ID, position, birthtime=0.0, proliferating=False):
        self.ID = ID
        self.position = position
        self.birthtime = birthtime
        self.proliferating = proliferating
        self.division_time = self.generate_division_time()

    def __lt__(self, other):
        return self.division_time < other.division_time

    def generate_division_time(self, current_time=None):
        current_time = current_time if current_time is not None else self.birthtime
        if self.proliferating:
            return np.random.randn() + 24.0 + current_time
        else:
            return np.inf
