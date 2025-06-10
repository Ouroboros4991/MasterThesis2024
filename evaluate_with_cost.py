import numpy as np
import pandas as pd

from sumo_rl_environment.custom_env import CustomSumoEnvironment
import torch

from agents import option_critic_nn

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
    curr_op_len = 1
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
    except Exception:
        greedy_option = 0
    while not terminate:
        obs = np.append(obs, curr_op_len)
        if option_termination:
            current_option = greedy_option
        try:
            action, _states = agent.predict(obs)
        except AttributeError:
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
        except Exception:
            pass

        if option_termination:
            curr_op_len = 1
        else:
            curr_op_len += 1

        cumulative_reward += rewards
        mean_waiting_time = info["system_mean_waiting_time"]
        mean_speed = info["system_mean_speed"]
        lane_density = np.sum(
            [ts.get_lanes_density() for ts in env.traffic_signals.values()]
        )
        queue_length = np.sum(
            [ts.get_total_queued() for ts in env.traffic_signals.values()]
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
                "queue_length": queue_length,
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
    results = []

    for episode_number in range(n_episodes):
        print("Episode", episode_number)
        episode_results = run_episode(env, agent)
        n_vehicles = float(
            env.sumo.simulation.getParameter(
                "", key="device.tripinfo.vehicleTripStatistics.count"
            )
        )
        total_travel_time = float(
            env.sumo.simulation.getParameter(
                "", key="device.tripinfo.vehicleTripStatistics.totalTravelTime"
            )
        )
        time_loss = float(
            env.sumo.simulation.getParameter(
                "", key="device.tripinfo.vehicleTripStatistics.timeLoss"
            )
        )
        waiting_time = float(
            env.sumo.simulation.getParameter(
                "", key="device.tripinfo.vehicleTripStatistics.waitingTime"
            )
        )

        collisions = float(
            env.sumo.simulation.getParameter("", key="stats.safety.collisions")
        )
        emergency_stops = float(
            env.sumo.simulation.getParameter("", key="stats.safety.emergencyStops")
        )
        emergency_braking = float(
            env.sumo.simulation.getParameter("", key="stats.safety.emergencyBraking")
        )

        avg_travel_time = total_travel_time / n_vehicles
        avg_time_loss = time_loss  # / n_vehicles
        avg_waiting_time = waiting_time  # / n_vehicles
        results.append(
            {
                "episode": episode_number,
                "cumulative_reward": episode_results[-1]["cumulative_reward"],
                "average_cumulative_reward": episode_results[-1][
                    "average_cumulative_reward"
                ],
                # Get the average waiting time across the episodes
                # "mean_waiting_time": np.mean(
                #     [r["waiting_time"] for r in episode_results]
                # ),
                # "mean_speed": np.mean([r["speed"] for r in episode_results]),
                "mean_lane_density": np.mean(
                    [r["lane_density"] for r in episode_results]
                ),
                "mean_queue_length": np.mean(
                    [r["queue_length"] for r in episode_results]
                ),
                "avg_travel_time": avg_travel_time,
                "avg_time_loss": avg_time_loss,
                "avg_waiting_time": avg_waiting_time,
                "collisions": collisions,
                "emergency_stops": emergency_stops,
                "emergency_braking": emergency_braking,
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
    env = CustomSumoEnvironment(
        net_file=route_file.format(type="net"),
        route_file=route_file.format(type="rou"),
        # single_agent=True,
        begin_time=start_time,
        num_seconds=duration,
    )

    # agent = default_4arm.CyclicAgent(env.action_space)
    # agent = stable_baselines3.PPO.load(
    #     "./models/ppo_custom-2way-single-intersection.zip"
    # )
    # agent = option_critic.OptionCriticFeatures(
    #     in_features=env.observation_space.shape[0] + 1,
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
    #         "./models/option_critic_2_options_with_deliberation_custom-2way-single-intersection_500000_steps"
    #     )["model_params"]
    # )

    agent = option_critic_nn.OptionCriticNeuralNetwork(
        in_features=env.observation_space.shape[0] + 1,
        num_actions=env.action_space.n,
        num_options=2,
        temperature=0.1,
        eps_start=0.9,
        eps_min=0.1,
        eps_decay=0.999,
        eps_test=0.05,
        device="cpu",
    )
    agent.load_state_dict(
        torch.load(
            "./models/option_critic_nn_2_options_with_deliberation_custom-2way-single-intersection_150000_steps"
        )["model_params"]
    )

    prefix = "oc_cost_150k_steps"
    single_episodes(env, agent, prefix)
    multiple_episodes(env, agent, prefix)
