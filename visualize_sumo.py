
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
        try:
            # env.env.sumo.vehicle.getLaneID is 0 if the vehicle is not in a lane yet
            print(env.env.sim_step, env.env.sumo.vehicle.getLaneID("emergency_0"), env.env.sumo.vehicle.getAccumulatedWaitingTime("emergency_0"))
        except Exception as e:
            print(e)
            pass

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
