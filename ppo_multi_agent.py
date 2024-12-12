import os
import sys


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import pandas as pd
import ray
import traci
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

import sumo_rl
from sumo_rl import SumoEnvironment
import gymnasium as gym

# import supersuit as ss
import stable_baselines3
from stable_baselines3.common.monitor import Monitor

# from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse

from configs import ROUTE_SETTINGS

TRAFFIC = "cologne3"  # "custom-2way-single-intersection"
SETTINGS = ROUTE_SETTINGS[TRAFFIC]
N_EPISODES = 250  # 100

if __name__ == "__main__":
    # Use:
    # ray[rllib]==2.7.0
    # numpy == 1.23.4
    # Pillow>=9.4.0
    # ray[rllib]==2.7.0
    # SuperSuit>=3.9.0
    # torch>=1.13.1
    # tensorflow-probability>=0.19.0

    route_file = SETTINGS["path"]
    start_time = SETTINGS["begin_time"]
    end_time = SETTINGS["end_time"]
    duration = end_time - start_time
    experiment_name = f"ppo_{TRAFFIC}"
    # delta_time (int) – Simulation seconds between actions. Default: 5 seconds
    single_agent = not (TRAFFIC in ("cologne3", "ingolstadt7"))

    print("Environment created")

    ray.init()

    register_env(
        TRAFFIC,
        lambda _: ParallelPettingZooEnv(
            sumo_rl.parallel_env(
                net_file=route_file.format(type="net"),
                route_file=route_file.format(type="rou"),
                # out_csv_name="outputs/4x4grid/ppo",
                use_gui=False,
                begin_time=start_time,
                num_seconds=duration,
                add_per_agent_info=True,
                add_system_info=True,
            )
        ),
    )

    config = (
        PPOConfig()
        .environment(env=TRAFFIC, disable_env_checking=True)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.95,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    # os.mkdir(f"./runs/{experiment_name}")
    tune.run(
        "PPO",
        name=experiment_name,
        stop={"timesteps_total": 1000000},
        checkpoint_freq=10,
        local_dir=f"~/runs/{experiment_name}",
        config=config.to_dict(),
    )
