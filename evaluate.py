import numpy as np
import pandas as pd
import operator

from sumo_rl import SumoEnvironment
import gymnasium as gym
import torch

import stable_baselines3
from agents import default_4arm
from agents import option_critic
from agents import option_critic_forced
from agents.option_critic_utils import to_tensor
from configs import ROUTE_SETTINGS

TRAFFIC = "custom-2way-single-intersection"
# TRAFFIC = "custom-single-intersection"
SETTINGS = ROUTE_SETTINGS[TRAFFIC]


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


def run_episode(env, agent):
    np.random.seed(42)
    torch.manual_seed(42)
    results = []

    obs, _ = env.reset()

    cumulative_reward = 0.0
    average_cumulative_reward = 0.0
    mean_waiting_time = 0.0
    mean_speed = 0.0

    terminate = False
    termination_prob = None
    option_termination = True
    try:
        state = agent.get_state(to_tensor(obs))
        greedy_option = agent.greedy_option(state)
    except Exception as e:
        greedy_option = 0
    while not terminate:
        if option_termination:
            current_option = greedy_option
        try:
            action, _states = agent.predict(obs)
        except AttributeError as e:
            # Option critic
            state = agent.get_state(to_tensor(obs))
            action, logp, entropy = agent.get_action(state, current_option)

        obs, rewards, dones, truncated, info = env.step(action)
        terminate = dones | truncated

        try:
            option_termination, greedy_option = agent.predict_option_termination(
                state, current_option
            )
            termination_prob = agent.get_terminations(state)[
                :, current_option
            ].tolist()[0]
        except Exception as e:
            pass

        cumulative_reward += rewards
        mean_waiting_time = info["system_mean_waiting_time"]
        mean_speed = info["system_mean_speed"]
        lane_density = np.sum(
            [ts.get_lanes_density() for ts in env.traffic_signals.values()]
        )
        average_cumulative_reward *= 0.95
        average_cumulative_reward += 0.05 * cumulative_reward

        results.append(
            {
                "step": info["step"],
                "option": current_option,
                "action": action,
                "obs": ", ".join([str(n) for n in obs]),
                "termination_prob": termination_prob,
                "should_terminate": option_termination,
                "greedy_option": greedy_option,
                "cumulative_reward": cumulative_reward,
                "average_cumulative_reward": average_cumulative_reward,
                # Get the average waiting time across the episodes
                "waiting_time": mean_waiting_time,
                "speed": mean_speed,
                "lane_density": lane_density,
            }
        )
    return results


def single_episodes(env, agent, prefix):
    results = run_episode(env, agent)
    print(f"./outputs/evaluation/{prefix}_1_episode_{TRAFFIC}.csv")
    pd.DataFrame(results).to_csv(
        f"./outputs/evaluation/{prefix}_1_episode_{TRAFFIC}.csv",
        index=False,
    )


def multiple_episodes(env, agent, prefix):
    n_episodes = 100
    average_cumulative_reward = 0.0
    results = []

    for episode_number in range(n_episodes):
        print("Episode", episode_number)
        episode_results = run_episode(env, agent)
        results.append(
            {
                "episode": episode_number,
                "cumulative_reward": episode_results[-1]["cumulative_reward"],
                "average_cumulative_reward": episode_results[-1][
                    "average_cumulative_reward"
                ],
                # Get the average waiting time across the episodes
                "mean_waiting_time": np.mean(
                    [r["waiting_time"] for r in episode_results]
                ),
                "mean_speed": np.mean([r["speed"] for r in episode_results]),
                "mean_lane_density": np.mean(
                    [r["lane_density"] for r in episode_results]
                ),
            }
        )
    print("Writing multiple episodes to csv")
    pd.DataFrame(results).to_csv(
        f"./outputs/evaluation/{prefix}_{n_episodes}_episode_{TRAFFIC}.csv",
        index=False,
    )


if __name__ == "__main__":
    route_file = SETTINGS["path"]
    start_time = SETTINGS["begin_time"]
    end_time = SETTINGS["end_time"]
    duration = end_time - start_time
    env = SumoEnvironment(
        net_file=route_file.format(type="net"),
        route_file=route_file.format(type="rou"),
        single_agent=True,
        use_gui=False,
        begin_time=start_time,
        num_seconds=duration,
        add_per_agent_info=True,
        add_system_info=True,
    )

    # agent = default_4arm.FourArmIntersection(env.action_space)
    agent = stable_baselines3.PPO.load("./models/ppo_custom-2way-single-intersection.zip")
    # agent = option_critic.OptionCriticFeatures(
    #     in_features=env.observation_space.shape[0],
    #     num_actions=env.action_space.n,
    #     num_options=2,
    #     temperature=0.1,
    #     eps_start=0.9,
    #     eps_min=0.1,
    #     eps_decay=0.999,
    #     eps_test=0.05,
    #     device="cpu",
    # )
    # agent.load_state_dict(
    #     torch.load(
    #         "./models/option_critic_2_options_custom-2way-single-intersection_500000_steps"
    #     )["model_params"]
    # )
    # agent = option_critic_forced.OptionCriticForced(
    #     in_features=env.observation_space.shape[0],
    #     num_actions=env.action_space.n,
    #     num_options=2,
    #     temperature=0.1,
    #     eps_start=0.9,
    #     eps_min=0.1,
    #     eps_decay=0.999,
    #     eps_test=0.05,
    #     device="cpu",
    # )
    # agent.load_state_dict(
    #     torch.load(
    #         "./models/option_critic_forced_2_options_custom-2way-single-intersection_500000_steps"
    #     )["model_params"]
    # )
    
    # agent = option_critic.OptionCriticFeatures(
    #     in_features=env.observation_space.shape[0],
    #     num_actions=env.action_space.n,
    #     num_options=2,
    #     temperature=0.1,
    #     eps_start=0.9,
    #     eps_min=0.1,
    #     eps_decay=0.999,
    #     eps_test=0.05,
    #     device="cpu",
    # )
    # agent.load_state_dict(
    #     torch.load(
    #         "./models/option_critic_2_options_custom-2way-single-intersection_hd_reg_500000_steps"
    #     )["model_params"]
    # )

    prefix = "ppo_500k_steps"
    single_episodes(env, agent, prefix)
    multiple_episodes(env, agent, prefix)
