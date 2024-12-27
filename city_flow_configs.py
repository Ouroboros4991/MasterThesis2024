"""File containing the configurations for the city flow simulation."""

import gymnasium as gym

from gym_cityflow.envs import CityflowGym
from gym_cityflow.envs import CityflowGymDiscrete

from city_flow_nets.real_1x1.config import CONFIG as REAL_1X1_CONFIG


gym.register(id="cityflow-base", entry_point="gym_cityflow.envs:CityflowGym")
gym.register(
    id="cityflow-discrete", entry_point="gym_cityflow.envs:CityflowGymDiscrete"
)
