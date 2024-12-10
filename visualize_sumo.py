import numpy as np
import pandas as pd
import operator

import gymnasium as gym
import torch

import stable_baselines3
from agents import default_4arm
from agents import option_critic
from agents.option_critic_utils import to_tensor
from configs import ROUTE_SETTINGS
from utils.custom_env import CustomSumoEnvironment


def visualize():
    # settings = ROUTE_SETTINGS["custom-2way-single-intersection"]
    settings = ROUTE_SETTINGS["cologne3"]
    route_file = settings["path"]
    start_time = settings["begin_time"]
    end_time = settings["end_time"]
    duration = end_time - start_time
    env = CustomSumoEnvironment(
        net_file=route_file.format(type="net"),
        route_file=route_file.format(type="rou"),
        single_agent=False,
        use_gui=True,
        begin_time=start_time,
        num_seconds=duration,
        delta_time=7,
    )
    agent = default_4arm.FourArmIntersection(env.action_space)
    # agent = stable_baselines3.PPO.load("./models/xj9bh2nc/model.zip")
    # agent = option_critic.OptionCriticFeatures(
    #     in_features=env.observation_space.shape[0],
    #     num_actions=env.action_space.n,
    #     num_options=2,
    #     temperature=0.1,
    #     eps_start=0.9,
    #     eps_min=0.1,
    #     eps_decay=0.999,
    #     eps_test=0.05,
    #     device="cpu",
    # )
    # agent.load_state_dict(
    #     torch.load(
    #         "./models/option_critic_2_options_custom-2way-single-intersection.csv_500000_steps"
    #     )["model_params"]
    # )
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
