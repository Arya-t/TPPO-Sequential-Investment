"""Minimal data structures and IO helpers for TPPO core pipeline.

This module intentionally keeps only the symbols required by:
- pickle loading of allarea_set objects (Region, AllArea)
- ROA.py wildcard import usage (np, random)
"""

import pickle
import random
import numpy as np


def save_variable_to_file(variable, filename):
    with open(filename, "wb") as file:
        pickle.dump(variable, file)


def load_variable_from_file(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


class Region:
    def __init__(self):
        self.id = 0
        self.layer = 1
        self.invest_state = False
        self.invest_time = 0
        self.od_demand = None
        self.d = 0
        self.mu = 0
        self.sigma = 0
        self.lambda_ = 0
        self.alpha = 0
        self.beta = 0
        self.area = 0
        self.density = 0
        self.adjacent_list = []


class AllArea:
    def __init__(self):
        self.t = 0
        self.invest_time = {}
        self.current_invest = {}
        self.region_dict = {}
        self.rewards = 0
