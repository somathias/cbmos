#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The average cell cycle duration for a typical rapidly proliferating human cell
is 24 hours (https://bionumbers.hms.harvard.edu/bionumber.aspx?id=112260).

As a default, we assume a normal distribution of cell cycle durations with
mean=24h and sigma=1.0h.
"""

import numpy as _np
import numpy.random as _npr


class Cell:
    """
    Parameters
    ----------
    ID: int
        Unique number used to identify the cell
    position : numpy array
        The position of this cell (1, 2, or 3d)
    birthtime: float
        Cell's birthtime (0 by default). The first division time will be
        computed from that value.
    proliferating: bool
        Whether or not the cell will proliferate
    division_time_generator : float -> float
        function that takes the current time to generate the absolute next
        division time.
    division_time: float
        first time at which the cell will divide next. If None, the division
        time generator will be use to set that time.
    parent_ID: int
        ID of the parent cell, can be used to reconstruct cell lineages.

    Note
    ----
    One should always use the constructor to access member variables,
    in order to ensure correct behavior. Eg. it is not possible to set the
    proliferating flag outside of the constructor because the division time
    would not be updated in that case.


    """
    def __init__(
            self, ID, position,
            birthtime=0.0,
            proliferating=False,
            division_time_generator=lambda t: _npr.normal(24 + t),
            division_time=None,
            parent_ID=None
            ):
        self.ID = ID
        self.position = _np.array(position)
        self.birthtime = birthtime
        self.proliferating = proliferating
        self.division_time = (
                division_time_generator(birthtime) if proliferating else _np.inf
                ) if division_time is None else division_time
        self.generate_division_time = division_time_generator
        self.parent_ID = parent_ID if parent_ID is not None else ID

    def __lt__(self, other):
        return self.division_time < other.division_time
