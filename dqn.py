import numpy as np
import operator
import wandb

import gymnasium as gym
import supersuit as ss
import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse

from gym_cityflow.envs import CityflowGym
from gym_cityflow.envs import CityflowGymDiscrete


import city_flow_configs as config


def main():
    experiment_name = f"dqn_real_1x1"
    env = gym.make(
        id="cityflow-discrete",
        config_dict=config.REAL_1X1_CONFIG,
        episode_steps=3600,  # TODO: remove episodeSteps and add it to the configDict,
    )
    print("Environment created")

    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class="stable_baselines3")
    # env = VecMonitor(env)

    env = Monitor(env)

    env.reset()

    agent = stable_baselines3.DQN(
        "MlpPolicy",
        env,
        verbose=3,
        tensorboard_log=f"runs/{experiment_name}",
    )

    agent.learn(total_timesteps=500000)
    agent.save(f"./models/{experiment_name}")


if __name__ == "__main__":
    main()
