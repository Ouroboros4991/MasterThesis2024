# Based on https://github.com/MaxVanDijck/gym-cityflow/blob/main/gym_cityflow/envs/cityflow_env.py

import json
import tempfile
import cityflow
import gymnasium as gym
import numpy as np
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

from gym_cityflow.envs.cityflow_base_env import CityflowGym


class CityflowGymDiscrete(CityflowGym):
    metadata = {"render.modes": ["human"]}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.intersections) != 1:
            raise ValueError(
                "CityflowGymDiscrete only supports single intersection scenarios"
            )
        self.main_intersection = list(self.intersections.keys())[0]
        self.action_space = self.action_space[self.main_intersection]

    def _generate_observation_space(self):
        """
        Generate the observation space for the environment
        """
        obs = self._get_observation()
        return spaces.MultiDiscrete([self.max_cars for _ in obs])

    def _set_tf_phases(self, action):
        """
        Set the traffic light phases of the main intersection
        """
        # As we only have 1 intersection,
        # we can directly set the traffic light phases
        self.eng.set_tl_phase(self.main_intersection, action)

    def _get_observation(self):
        main_intersection = list(self.intersections.keys())[0]
        obs = super()._get_observation()[main_intersection]
        return obs
