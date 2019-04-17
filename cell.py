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
    def __init__(self, ID, position, age=0.0, proliferating=False):
        self.ID = ID
        self.position = position
        self.age = age
        self.proliferating = proliferating
        self.division_time = self.generate_division_time()

    def generate_division_time(self):
        if self.proliferating:
            return np.random.randn() + 24.0 - self.age
        else:
            return None
