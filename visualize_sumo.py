
import argparse
import numpy as np
import pandas as pd
import operator

import gymnasium as gym
import torch

from agents.option_critic_utils import to_tensor
from configs import ROUTE_SETTINGS
from sumo_rl_environment.custom_env import CustomSumoEnvironment, BrokenLightEnvironment
from utils import utils

def visualize(traffic: str, model: str):
    settings = ROUTE_SETTINGS[traffic]
    route_file = settings["path"]
    start_time = settings["begin_time"]
    end_time = settings["end_time"]
    duration = end_time - start_time
    env = BrokenLightEnvironment(  # TODO: add variable to change this
        net_file=route_file.format(type="net"),
        route_file=route_file.format(type="rou"),
        use_gui=True,
        begin_time=start_time,
        num_seconds=duration,
        reward_fn="intelli_light_prcol_reward",
        # broken_light_start=100,
        # broken_light_end=500
        
    )
    if model.startswith("a2c"):
        env = utils.DictToFlatActionWrapper(env)
    obs, _ = env.reset()

    agent = utils.load_model(model, env)

    terminate = False
    option_termination = True
    try:
        state = agent.get_state(to_tensor(obs))
        greedy_option = agent.greedy_option(state)
    except Exception as e:
        greedy_option = 0
    while not terminate:
        try:
            action_dict, _states = agent.predict(obs)
            action = action_dict
        except AttributeError as e:
            # Option critic
            state = agent.prep_state(obs)
            action, additional_info = agent.get_action(state)
            action_dict = agent.convert_action_to_dict(action)
        # print("Action dict:", action_dict)
        obs, reward, terminate, truncated, info = env.step(action_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    description='Evaluate the provided model using the given traffic scenario.',
                    )
    parser.add_argument('-m', '--model', required=True) 
    parser.add_argument('-t', '--traffic',
                        choices=ROUTE_SETTINGS.keys(),
                        required=True) 
    args = parser.parse_args()
    visualize(args.traffic, args.model)
