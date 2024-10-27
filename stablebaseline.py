import numpy as np
import operator
import wandb

from sumo_rl import SumoEnvironment
import gymnasium as gym
import supersuit as ss
import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.vec_env import VecMonitor

import argparse

routes = {
    "single-intersection": "sumo_rl/nets/single-intersection/single-intersection.{type}.xml",
    "ingolstadt7": "sumo_rl/nets/RESCO/ingolstadt7/ingolstadt7.{type}.xml",
}

agents = {
    "ppo": stable_baselines3.PPO,
    # "a2c": stable_baselines3.A2C,
}


def main():
    experiment_name = "_".join([f"{key}_{value}" for key, value in vars(args).items()])
    route_file = routes[args.route]
    run = wandb.init(
        project="thesis",
        config=args,
        sync_tensorboard=True,
    )

    env = SumoEnvironment(
        net_file=route_file.format(type="net"),
        route_file=route_file.format(type="rou"),
        out_csv_name=f"./outputs/{experiment_name}.csv",
        single_agent=True,
        begin_time=57600,
        num_seconds=61200 - 57600,
    )

    print("Environment created")

    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class="stable_baselines3")
    # env = VecMonitor(env)

    env = Monitor(env)

    env.reset()

    agent = agents[args.agent](
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
        tensorboard_log=f"runs/{run.id}",
    )

    agent.learn(
        total_timesteps=args.timesteps,
        callback=WandbCallback(
            gradient_save_freq=1,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()


def argumentParser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument('--epsilon', default=0.05, type=float, help='Probability of chossing random action')
    # parser.add_argument('--alpha', default=0.1, type=float, help='Learning Rate')
    # parser.add_argument('--gamma', default=0.95, type=float, help='Discounting Factor')
    parser.add_argument(
        "--agent", required=True, type=str, help="Agent to use", choices=agents.keys()
    )
    parser.add_argument(
        "--route", required=True, type=str, help="Route file", choices=routes.keys()
    )
    parser.add_argument("--timesteps", required=True, type=int)

    return parser


if __name__ == "__main__":
    global args
    args = argumentParser().parse_args()
    main()
