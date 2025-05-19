import numpy as np
import operator
import json
# import wandb

from sumo_rl import SumoEnvironment
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

# import supersuit as ss
import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse

from sumo_rl_environment.custom_env import CustomSumoEnvironment, BrokenLightEnvironment

from configs import ROUTE_SETTINGS
from utils.utils import DictToFlatActionWrapper


def setup_env(traffic: str, reward_fn: str, reward_weights: dict = {}, broken: bool = False):
    """Setup the environment for the given traffic scenario.

    Args:
        traffic (str): traffic to train on
    
    Returns:
        env: the environment
    """
    settings = ROUTE_SETTINGS[traffic]
    route_file = settings["path"]
    start_time = settings["begin_time"]
    end_time = settings["end_time"]
    duration = end_time - start_time

    # delta_time (int) – Simulation seconds between actions. Default: 5 seconds
    if broken:
        env = BrokenLightEnvironment(
            net_file=route_file.format(type="net"),
            route_file=route_file.format(type="rou"),
            # single_agent=True,
            begin_time=start_time,
            num_seconds=duration,
            reward_fn=reward_fn,
            intelli_light_weight=reward_weights,
        )
    else:
        env = CustomSumoEnvironment(
            net_file=route_file.format(type="net"),
            route_file=route_file.format(type="rou"),
            # single_agent=True,
            begin_time=start_time,
            num_seconds=duration,
            reward_fn=reward_fn,
            intelli_light_weight=reward_weights,
        )

    print("Environment created")
    env = DictToFlatActionWrapper(env) 

    env = Monitor(env)

    # env = FlattenObservation(env)
    print(env.action_space, env.observation_space)
    return env


def train(env, traffic: str, steps: int = 30000, reward_fn: str = "pressure", broken: bool = False):
        
    if broken:
        experiment_name = f"a2c_broken_{traffic}_{steps}_steps"
    else:
        experiment_name = f"a2c_{traffic}_{steps}_steps"
    experiment_name += reward_fn
    if env.env.intelli_light_weight:
        experiment_name += "_".join([f"{key}_{value}" for key, value in env.env.intelli_light_weight.items()])
    # env = DummyVecEnv([lambda: env])
    env.reset()

    agent = stable_baselines3.A2C(
        "MultiInputPolicy",
        env,
        verbose=3,
        gamma=0.95,
        tensorboard_log=f"runs/{experiment_name}",
        # learning_rate=0.0001,
        ent_coef=0.02
    )

    agent.learn(total_timesteps=steps)
    agent.save(f"models/{experiment_name}.zip")
    return agent


def main(traffic: str, steps: int, reward_fn: str, broken: bool = False):
    """Main function to train the agent."""
    
    weights = {
        "delay": 3,
        "waiting_time": 3,
        "light_switches": 2,
        "out_lanes_availability": 1
    }
    
    env = setup_env(traffic, reward_fn, reward_weights=weights, broken=broken)
    train(env, traffic, steps, reward_fn, broken)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    description='Evaluate the provided model using the given traffic scenario.',
                    )
    parser.add_argument('-t', '--traffic',
                        choices=ROUTE_SETTINGS.keys(),
                        required=True) 
    parser.add_argument('-s', '--steps',
                        type=int,
                        default=100000,
                        help='Number of steps to train the agent.')
    parser.add_argument('-b', '--broken',
                        action='store_true',
                        help='Use broken traffic lights.')
    parser.add_argument('-r', "--reward_fn",
                        type=str,
                        help="Reward function to use",
                        default="intelli_light_reward"
                        )
    args = parser.parse_args()
    # reward_fn = "intelli_light_prcol_reward"

    main(args.traffic, args.steps, args.reward_fn, args.broken)
