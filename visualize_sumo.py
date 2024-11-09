import numpy as np
import operator

from sumo_rl import SumoEnvironment
import gymnasium as gym

import stable_baselines3
from agents import default_4arm

from configs import ROUTE_SETTINGS


def visualize(model: str):
    settings = ROUTE_SETTINGS["cologne1"]
    route_file = settings["path"]
    start_time = settings["begin_time"]
    end_time = settings["end_time"]
    
    duration = end_time - start_time
    env = SumoEnvironment(
        net_file=route_file.format(type="net"),
        route_file=route_file.format(type="rou"),
        single_agent=True,
        use_gui=True,
        begin_time=start_time,
        num_seconds=duration,
    )
    # model = stable_baselines3.PPO.load("./models/xj9bh2nc/model.zip")
    agent = default_4arm.FourArmIntersection()

    obs, _ = env.reset()
    terminate = False
    while not terminate:
        action, _states = agent.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        terminate = dones | truncated


if __name__ == "__main__":
    visualize("test")
