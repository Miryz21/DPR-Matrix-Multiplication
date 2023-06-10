import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TensorEnv(gym.Env):

    def __init__(self):
        self.space = 4

        self.action_space = spaces.Discrete(self.space*2*3)

        self._action_to_vector = {}

        for i in range(self.space*2*3):
            


