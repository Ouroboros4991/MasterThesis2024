import numpy as np
import operator
import wandb

from sumo_rl import SumoEnvironment
import gymnasium as gym

import stable_baselines3
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

routes = {
    "single-intersection": "sumo_rl/nets/single-intersection/single-intersection.{type}.xml",
}


def visualize(model: str):
    route_file = routes["single-intersection"]
    env = SumoEnvironment(
        net_file=route_file.format(type="net"),
        route_file=route_file.format(type="rou"),
        single_agent=True,
        use_gui=True,
    )
    model = stable_baselines3.PPO.load("./models/xj9bh2nc/model.zip")

    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)


if __name__ == "__main__":
    visualize("test")
