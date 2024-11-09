import numpy as np
import pandas as pd
import operator

from sumo_rl import SumoEnvironment
import gymnasium as gym

import stable_baselines3
from agents import default_4arm

from configs import ROUTE_SETTINGS

MODEL = 'custom-single-intersection'
SETTINGS = ROUTE_SETTINGS[MODEL]


# Example of the info payload
# {
#     'step': 28660.0,
#     'system_total_stopped': 4,
#     'system_total_waiting_time': 15.0,
#     'system_mean_waiting_time': 0.5555555555555556,
#     'system_mean_speed': 8.407979425401129,
#     'GS_cluster_357187_359543_stopped': 4,
#     'GS_cluster_357187_359543_accumulated_waiting_time': 16.0,
#     'GS_cluster_357187_359543_average_speed': 0.2459983516332396,
#     'agents_total_stopped': 4,
#     'agents_total_accumulated_waiting_time': 16.0
#  }

def single_episodes():    
    route_file = SETTINGS["path"]
    start_time = SETTINGS["begin_time"]
    end_time = SETTINGS["end_time"]   
    duration = end_time - start_time

    average_cumulative_reward = 0.0
    results = []
    env = SumoEnvironment(
        net_file=route_file.format(type="net"),
        route_file=route_file.format(type="rou"),
        single_agent=True,
        use_gui=False,
        begin_time=start_time,
        num_seconds=duration,
        add_per_agent_info=True,
        add_system_info=True
    )
    # model = stable_baselines3.PPO.load("./models/xj9bh2nc/model.zip")
    agent = default_4arm.FourArmIntersection(env.action_space)

    obs, _ = env.reset()
    
    cumulative_reward = 0.0
    mean_waiting_time = 0.0
    mean_speed = 0.0
    queue_length = 0.0
    
    terminate = False
    while not terminate:
        action, _states = agent.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        terminate = dones | truncated

        cumulative_reward += rewards
        mean_waiting_time += info['system_mean_waiting_time']
        mean_speed += info['system_mean_speed']
        
            
        average_cumulative_reward *= 0.95
        average_cumulative_reward += 0.05 * cumulative_reward
        # print(algorithm.epsilon)
        
        lane_density = sum(obs[2: len(obs-2)//2])
        
        results.append({
            'step': info['step'],
            'cumulative_reward': cumulative_reward,
            'average_cumulative_reward': average_cumulative_reward,
            
            # Get the average waiting time across the episodes
            'mean_waiting_time': mean_waiting_time,
            'mean_speed': mean_speed,
            'queue_length': lane_density,
        })
    pd.DataFrame(results).to_csv(f'./outputs/sumo_exploration/sumo_exploration_1_episode_{MODEL}.csv', index=False)



def multiple_episodes():
    n_episodes = 100
    
    route_file = SETTINGS["path"]
    start_time = SETTINGS["begin_time"]
    end_time = SETTINGS["end_time"]   
    duration = end_time - start_time

    average_cumulative_reward = 0.0
    results = []
    env = SumoEnvironment(
        net_file=route_file.format(type="net"),
        route_file=route_file.format(type="rou"),
        single_agent=True,
        use_gui=False,
        begin_time=start_time,
        num_seconds=duration,
        add_per_agent_info=True,
        add_system_info=True
    )
    agent = default_4arm.FourArmIntersection(env.action_space)

    for e in range(n_episodes):
        # model = stable_baselines3.PPO.load("./models/xj9bh2nc/model.zip")

        obs, _ = env.reset()
        
        cumulative_reward = 0.0
        mean_waiting_time = 0.0
        mean_speed = 0.0
        lane_density = 0.0
        terminate = False
        while not terminate:
            action, _states = agent.predict(obs)
            obs, rewards, dones, truncated, info = env.step(action)
            terminate = dones | truncated

            cumulative_reward += rewards
            mean_waiting_time += info['system_mean_waiting_time']
            mean_speed += info['system_mean_speed']
            lane_density += sum(obs[2: len(obs-2)//2])
            
        average_cumulative_reward *= 0.95
        average_cumulative_reward += 0.05 * cumulative_reward
        print('Episode', e)
        results.append({
            'episode': e,
            'cumulative_reward': cumulative_reward,
            'average_cumulative_reward': average_cumulative_reward,
            
            # Get the average waiting time across the episodes
            'mean_waiting_time': mean_waiting_time/ duration,
            'mean_speed': mean_speed / duration,
            'mean_lane_density': lane_density/ duration,
        })
    print('Writing multiple episodes to csv')
    pd.DataFrame(results).to_csv(f'./outputs/sumo_exploration/sumo_exploration_{n_episodes}_episode_{MODEL}.csv', index=False)

        


if __name__ == "__main__":
    single_episodes()
    multiple_episodes()