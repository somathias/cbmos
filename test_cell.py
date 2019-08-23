#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:39:37 2019

@author: Sonja Mathias
"""
import cell as cl


def test_parameters():

    ID = 17
    position = [17.4]
    birthtime = 0.45
    proliferating = True
    division_time = 5.89
    parent_ID = 5

    cell = cl.Cell(ID, position, birthtime, proliferating, division_time,
                   parent_ID)

    assert cell.ID == ID
    assert cell.position == position
    assert cell.birthtime == birthtime
    assert cell.proliferating == proliferating
    assert cell.division_time == division_time
    assert cell.parent_ID == parent_ID
