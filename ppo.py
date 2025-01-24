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

TRAFFIC = "cologne1" #"custom-2way-single-intersection"
SETTINGS = ROUTE_SETTINGS[TRAFFIC]
N_EPISODES = 100


def main():
    route_file = SETTINGS["path"]
    start_time = SETTINGS["begin_time"]
    end_time = SETTINGS["end_time"]
    duration = end_time - start_time
    experiment_name = f"ppo_1mil_{TRAFFIC}"
    # delta_time (int) – Simulation seconds between actions. Default: 5 seconds
    env = CustomSumoEnvironment(
        net_file=route_file.format(type="net"),
        route_file=route_file.format(type="rou"),
        # single_agent=True,
        begin_time=start_time,
        num_seconds=duration,
    )
    print("Environment created")

    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class="stable_baselines3")
    # env = VecMonitor(env)

    env = Monitor(env)

    env.reset()

    agent = stable_baselines3.PPO(
        "MlpPolicy",
        env,
        verbose=3,
        gamma=0.95,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=0.00062211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=5,
        clip_range=0.3,
        batch_size=256,
        tensorboard_log=f"runs/{experiment_name}",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1000000,
        save_path="./models/",
        name_prefix=experiment_name,
    )

    agent.learn(total_timesteps=1000000)
    agent.save(f"./models/{experiment_name}")


if __name__ == "__main__":
    main()
