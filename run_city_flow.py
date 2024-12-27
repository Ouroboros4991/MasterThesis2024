import numpy as np
import pandas as pd
import operator

import gymnasium as gym
import torch

import stable_baselines3
from agents import fixedtime_agent
from agents import option_critic
from agents.option_critic_utils import to_tensor

from gym_cityflow.envs import CityflowGym
from gym_cityflow.envs import CityflowGymDiscrete

from city_flow_nets.real_1x1 import config
import city_flow_configs as config


def visualize():
    env = gym.make(
        id="cityflow-discrete",
        config_dict=config.REAL_1X1_CONFIG,
        episode_steps=3600,  # TODO: remove episodeSteps and add it to the configDict
    )

    agent = fixedtime_agent.FixedTimeAgent(
        env=env,
        n_actions=env.action_space.n,
        rotation_period=30,
    )
    # agent = stable_baselines3.PPO.load("./models/ppo_real_1x1.zip")
    # obs, _ = env.reset()
    # terminate = False
    # while not terminate:
    #     action = env.action_space.sample()
    #     action = np.array([action])
    #     obs, rewards, dones, truncated, info = env.step(action)
    #     terminate = dones | truncated

    obs, _ = env.reset()
    terminate = False
    option_termination = True
    try:
        state = agent.get_state(to_tensor(obs))
        greedy_option = agent.greedy_option(state)
    except Exception as e:
        greedy_option = 0
    while not terminate:
        if option_termination:
            current_option = greedy_option

        try:
            action, _states = agent.predict(obs)
        except AttributeError as e:
            state = agent.get_state(to_tensor(obs))
            action, logp, entropy = agent.get_action(state, current_option)
        obs, rewards, dones, truncated, info = env.step(action)
        terminate = dones | truncated
        try:
            option_termination, greedy_option = agent.predict_option_termination(
                state, current_option
            )
        except Exception as e:
            pass


if __name__ == "__main__":
    visualize()
