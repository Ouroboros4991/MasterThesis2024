
import argparse
import numpy as np
import pandas as pd
import operator

import gymnasium as gym
import torch

from agents.option_critic_utils import to_tensor
from configs import ROUTE_SETTINGS
from sumo_rl_environment.custom_env import CustomSumoEnvironment
from utils import utils

def visualize(traffic: str, model: str):
    settings = ROUTE_SETTINGS[traffic]
    route_file = settings["path"]
    start_time = settings["begin_time"]
    end_time = settings["end_time"]
    duration = end_time - start_time
    env = CustomSumoEnvironment(
        net_file=route_file.format(type="net"),
        route_file=route_file.format(type="rou"),
        use_gui=True,
        begin_time=start_time,
        num_seconds=duration,
    )
    obs = env.reset()

    agent = utils.load_model(model, env)

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
        
        obs, rewards, dones, info = env.step(action)
        terminate = dones['__all__']
        try:
            option_termination, greedy_option = agent.predict_option_termination(
                state, current_option
            )
        except Exception as e:
            pass


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
