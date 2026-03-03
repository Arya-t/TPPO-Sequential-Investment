# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:41:27 2024

@author: dingye
"""

import gym
from gym import spaces
from ROA import *
import itertools
import pickle
import matplotlib.pyplot as plt
import arviz as az
import scipy.stats as stats
import os
import numpy as np


class InvestEnv_Train(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, num_regions, allarea_set, roa):
        self.num_regions = num_regions
        self.roa = roa
        self.action_space = spaces.MultiBinary(self.num_regions)

        # state = [t, invested_ratio, invest_state(N), initial_demand_norm(N)]
        self.observation_space = spaces.Box(
            low=np.concatenate([[0], [0], np.zeros(self.num_regions), np.zeros(self.num_regions)]),
            high=np.concatenate([[roa.T], [1], np.ones(self.num_regions), np.ones(self.num_regions)]),
            dtype=np.float64
        )

        self.reset(allarea_set)

    def reset(self, allarea_set=None):
        if allarea_set is not None:
            self.allarea_set = copy.deepcopy(allarea_set)

        area = self.allarea_set[0]
        self.sequence_position = 0
        self.region_dict = area.region_dict
        self.invest_state = np.zeros(self.num_regions, dtype=int)
        self.current_invest = area.current_invest
        self.t = 0
        self.invested_regions = []
        self.invest_sequence = []

        demand_raw = np.array(
            [max(0.0, float(self.region_dict[i + 1].d)) for i in range(self.num_regions)],
            dtype=np.float64
        )
        d_min = float(np.min(demand_raw))
        d_max = float(np.max(demand_raw))
        if d_max > d_min:
            self.initial_demand_norm = (demand_raw - d_min) / (d_max - d_min)
        else:
            self.initial_demand_norm = np.zeros_like(demand_raw)

        return self._get_observation()

    def step(self, action):
        portfolio = []
        for i in range(self.num_regions):
            region_id = i + 1
            if action[i] > 0.5:
                self.region_dict[region_id].invest_state = True
                self.current_invest[region_id] = 1
                portfolio.append(region_id)
                self.invest_state[i] = 1

        reward = 0
        if len(portfolio) > 0:
            self.invest_sequence.append(portfolio)
            for region_id in portfolio:
                self.invested_regions.append(region_id)

        self.sequence_position += 1

        self.t += 1
        done = self.t > self.roa.T
        observation = self._get_observation()
        area = self.allarea_set[0]
        self.current_invest = area.current_invest
        return observation, reward, done, {}

    def generate_mask(self):
        mask = [0 if region.invest_state else 1 for region in self.region_dict.values()]
        return mask

    def _get_observation(self):
        invested_ratio = sum(self.invest_state) / self.num_regions
        t_feature = min(self.t, self.roa.T)
        return np.concatenate([
            [t_feature],
            [invested_ratio],
            self.invest_state,
            self.initial_demand_norm,
        ])

    def render(self, mode='human'):
        return None

    def close(self):
        return None
