#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:18:39 2019

@author: kubuntu1804
"""
import force_functions as ff

def test_hertz():
    assert ff.hertz(1.0) == 0.
    assert ff.hertz(5/9) ==-8/27