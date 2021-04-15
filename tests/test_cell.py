#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:39:37 2019

@author: Sonja Mathias
"""
import cbmos.cell as cl


def test_parameters():

    ID = 17
    position = [17.4]
    birthtime = 0.45
    proliferating = True
    division_time_generator = lambda t: 3
    parent_ID = 5

    cell = cl.Cell(ID, position, birthtime, proliferating, division_time_generator,
                   None, parent_ID)

    assert cell.ID == ID
    assert cell.position == position
    assert cell.birthtime == birthtime
    assert cell.proliferating == proliferating
    assert cell.division_time == division_time_generator(birthtime)
    assert cell.parent_ID == parent_ID
