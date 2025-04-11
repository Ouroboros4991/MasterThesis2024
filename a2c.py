import numpy as np
import operator

# import wandb

from sumo_rl import SumoEnvironment
import gymnasium as gym

# import supersuit as ss
import stable_baselines3
from stable_baselines3.common.monitor import Monitor

# from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse

from sumo_rl_environment.custom_env import CustomSumoEnvironment

from configs import ROUTE_SETTINGS
from utils.utils import DictToFlatActionWrapper


def setup_env(traffic: str, reward_fn: str, reward_weights: dict = {}):
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
    return DictToFlatActionWrapper(env) 


def train(env, traffic: str, steps: int = 30000):    
    experiment_name = f"a2c_{traffic}_{steps}_steps"

    env = Monitor(env)

    env.reset()

    agent = stable_baselines3.A2C(
        "MultiInputPolicy",
        env,
        verbose=3,
        gamma=0.95,
        tensorboard_log=f"runs/{experiment_name}",
    )

    agent.learn(total_timesteps=steps)
    agent.save(f"models/{experiment_name}.zip")
    return agent

def main():
    # training_steps = 50000
    training_steps = 100000
    traffic_low = "custom-2way-single-intersection-low"
    traffic_high = "custom-2way-single-intersection-high"
    
    weights = {"delay": 3, "waiting_time": 3, "light_switches": 2}
    
    env_low = setup_env(traffic_low, "intelli_light_reward", reward_weights=weights)
    agent_low = train(env_low, traffic_low, training_steps)

    env_high = setup_env(traffic_high, "intelli_light_reward", reward_weights=weights)
    agent_high = train(env_high, traffic_high, training_steps)

if __name__ == "__main__":
    main()
