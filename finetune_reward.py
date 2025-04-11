
import numpy as np
import argparse
import torch
from copy import deepcopy
import itertools
import math
import pandas as pd
import json

import gymnasium as gym
from gymnasium.spaces import Dict, MultiDiscrete, Discrete

from agents.actor_critic_agent import CustomActorCritic

from sumo_rl_environment.custom_env import CustomSumoEnvironment
import stable_baselines3
from stable_baselines3.common.monitor import Monitor


from utils.sb3_logger import SB3Logger as Logger
from utils.utils import DictToFlatActionWrapper

import time

from configs import ROUTE_SETTINGS
import evaluate


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
    experiment_name = "finetuning_reward_a2c"

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
    return agent


def __get_action_distribution(obs, policy_model):
    """Get the action distribution for the given state and option.

    Args:
        state: State to calculate the action distribution for.
        model: Model to calculate the action distribution with.
        option: Option to calculate the action distribution for.

    Returns:
        array: Numpy array with the action distribution.
    """
    state = {}
    for key, observation_dict in obs.items():
        phase_ids = observation_dict["hist_phase_ids"]
        queue = observation_dict["queue"]
        queue_der = observation_dict["queue_der"]
        average_speed = observation_dict["average_speed"]
        waiting_times = []
        for waiting_time in observation_dict["waiting_times"]:
            if waiting_time:
                waiting_times.append(np.mean(waiting_time))
            else:
                waiting_times.append(0)
        combined_obs = np.array(phase_ids + queue + queue_der + waiting_times + [average_speed], dtype=np.float32)
        state[key] = torch.tensor([combined_obs], dtype=torch.float32, device="cuda")
    return policy_model.get_distribution(state).distribution[0].probs.cpu().detach().numpy()[0]


def hellinger_distance(state, model_1, model_2) -> float:
    """Calculate the hellinger distance between the intra-option policies of the model
    for the given state.
    This as defined in the paper "Disentangling Options with Hellinger Distance Regularizer"

    Args:
        states: The states for which to calculate the hellinger distance.
        model: The option critic model to calculate the distance for.

    Returns:
        float: Helling distance loss
    """
    p_dist = __get_action_distribution(state, model_1.policy)
    q_dist = __get_action_distribution(state, model_2.policy)
    summation = np.sum((np.sqrt(p_dist) - np.sqrt(q_dist)) ** 2)
    hd = math.sqrt(summation) / math.sqrt(2)
    return hd


def eval_episode(eval_env_low, eval_env_high, agent_low, agent_high):
    single_episode_result = evaluate.single_episodes(env=eval_env_low, agent=agent_low, prefix="finetuning_low_low", save=False)
    hd_arr = [
        hellinger_distance(json.loads(item["obs"]), agent_low, agent_high)
        for item in single_episode_result
    ]
    
    reward_arr = [
        item["cumulative_reward"] for item in single_episode_result
    ]
    avg_hd_distance_low_low = np.mean(hd_arr)
    avg_reward_low_low = np.mean(reward_arr)
    
    
    single_episode_result = evaluate.single_episodes(env=eval_env_low, agent=agent_high, prefix="finetuning_low_high", save=False)
    hd_arr = [
        hellinger_distance(json.loads(item["obs"]), agent_low, agent_high)
        for item in single_episode_result
    ]
    
    reward_arr = [
        item["cumulative_reward"] for item in single_episode_result
    ]
    avg_hd_distance_low_high = np.mean(hd_arr)
    avg_reward_low_high = np.mean(reward_arr)
    
    
    single_episode_result = evaluate.single_episodes(env=eval_env_high, agent=agent_low, prefix="finetuning_high_low", save=False)
    hd_arr = [
        hellinger_distance(json.loads(item["obs"]), agent_low, agent_high)
        for item in single_episode_result
    ]
    
    reward_arr = [
        item["cumulative_reward"] for item in single_episode_result
    ]
    avg_hd_distance_high_low = np.mean(hd_arr)
    avg_reward_high_low = np.mean(reward_arr)
    
    
    single_episode_result = evaluate.single_episodes(env=eval_env_high, agent=agent_high, prefix="finetuning_high_high", save=False)
    hd_arr = [
        hellinger_distance(json.loads(item["obs"]), agent_low, agent_high)
        for item in single_episode_result
    ]
    
    reward_arr = [
        item["cumulative_reward"] for item in single_episode_result
    ]
    avg_hd_distance_high_high = np.mean(hd_arr)
    avg_reward_high_high = np.mean(reward_arr)
    return {
        "avg_hd_distance_low_low": avg_hd_distance_low_low,
        "avg_reward_low_low": avg_reward_low_low,
        "avg_hd_distance_low_high": avg_hd_distance_low_high,
        "avg_reward_low_high": avg_reward_low_high,
        "avg_hd_distance_high_low": avg_hd_distance_high_low,
        "avg_reward_high_low": avg_reward_high_low,
        "avg_hd_distance_high_high": avg_hd_distance_high_high,
        "avg_reward_high_high": avg_reward_high_high,
    }

def generate_possible_reward_weights():
    reward_weight_values = {
        # 'queue': [0,1,2,3],
        'delay': [0,1,2,3],
        'waiting_time': [0,1,2,3],
        'light_switches': [0,1,2,3],
    }
    reward_configs = []
    # for queue_weight in reward_weight_values['queue']:
    for delay_weight in reward_weight_values['delay']:
        for waiting_time_weight in reward_weight_values['waiting_time']:
            for light_switches_weight in reward_weight_values['light_switches']:
                reward_configs.append({
                    # 'queue': queue_weight,
                    'delay': delay_weight,
                    'waiting_time': waiting_time_weight,
                    'light_switches': light_switches_weight,
                })
    return reward_configs


def main():
    already_run = pd.read_csv("./evaluations/finetuning_overview_doubling_max_lane_1.csv")
    known_weights = [json.loads(item) for item in already_run.config.to_list()]
    final_results = []
    reward_weights = generate_possible_reward_weights()
    for weights in reward_weights:
        if weights in known_weights:
            print(f"Already run {weights}")
            continue
        print(f"TESTING FOR {weights}")
        training_steps = 30000
        traffic_low = "custom-2way-single-intersection-low"
        traffic_high = "custom-2way-single-intersection-high"
        
        env_low = setup_env(traffic_low, "intelli_light_reward", reward_weights=weights)
        agent_low = train(env_low, traffic_low, training_steps)

        env_high = setup_env(traffic_high, "intelli_light_reward", reward_weights=weights)
        agent_high = train(env_high, traffic_high, training_steps)


        cross_episode_results = []
        eval_env_low = setup_env(traffic_low, "pressure")
        eval_env_high = setup_env(traffic_high, "pressure")
        for _ in range(10):
            try:
                result_dict = eval_episode(eval_env_low, eval_env_high, agent_low, agent_high)
                cross_episode_results.append(result_dict)
            except Exception:
                result_dict = eval_episode(eval_env_low, eval_env_high, agent_low, agent_high)
                cross_episode_results.append(result_dict)
        avg_result = {
            "config": json.dumps(weights),
            "avg_hd_distance_low_low": np.mean([item["avg_hd_distance_low_low"] for item in cross_episode_results]),
            "avg_reward_low_low": np.mean([item["avg_reward_low_low"] for item in cross_episode_results]),
            "avg_hd_distance_low_high": np.mean([item["avg_hd_distance_low_high"] for item in cross_episode_results]),
            "avg_reward_low_high": np.mean([item["avg_reward_low_high"] for item in cross_episode_results]),
            "avg_hd_distance_high_low": np.mean([item["avg_hd_distance_high_low"] for item in cross_episode_results]),
            "avg_reward_high_low": np.mean([item["avg_reward_high_low"] for item in cross_episode_results]),
            "avg_hd_distance_high_high": np.mean([item["avg_hd_distance_high_high"] for item in cross_episode_results]),
            "avg_reward_high_high": np.mean([item["avg_reward_high_high"] for item in cross_episode_results]),
        }
        final_results.append(avg_result)
        print(avg_result)
        print("=========================")

        pd.DataFrame(final_results).to_csv(
            f"./evaluations/finetuning_overview_doubling_max_lane.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
