"""Evaluate the provided model using the given environment."""

# TODO: add logic to

import json
import pathlib

import numpy as np
import pandas as pd
import torch

from configs import ROUTE_SETTINGS
from agents import option_critic_nn
from sumo_rl_environment.custom_env import CustomSumoEnvironment

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
    # np.random.seed(42)
    # torch.manual_seed(42)
    results = []
    num_options = 2

    obs = env.reset()
    current_option = 0

    traffic_light_id = list(obs.keys())[0]

    # TODO: update for multi agent setup
    obs = obs[traffic_light_id]
    encoded_option = np.zeros(num_options)
    encoded_option[current_option] = 1
    obs = np.append(obs, encoded_option)

    cumulative_reward = 0.0
    average_cumulative_reward = 0.0
    mean_waiting_time = 0.0
    mean_speed = 0.0

    terminate = False
    termination_prob = None
    option_termination = True
    state = agent.get_state(obs)
    greedy_option = agent.greedy_option(state)
    while not terminate:
        if option_termination:
            current_option = greedy_option

        state = agent.get_state(obs)
        action, logp, entropy = agent.get_action(state, current_option)

        next_obs, reward, dones, info = env.step({traffic_light_id: action})
        terminate = dones["__all__"]
        reward = reward[traffic_light_id]
        next_obs = next_obs[traffic_light_id]
        encoded_option = np.zeros(num_options)
        encoded_option[current_option] = 1
        next_obs = np.append(next_obs, encoded_option)

        state = agent.get_state(next_obs)
        # TODO: include min length
        option_termination, greedy_option = agent.predict_option_termination(
            state, current_option
        )

        termination_prob = agent.get_terminations(state)[:, current_option].tolist()[0]

        reward_arr = [reward]
        cumulative_reward += np.mean(reward_arr)
        mean_waiting_time = info["system_mean_waiting_time"]
        mean_speed = info["system_mean_speed"]
        lane_density = np.sum(
            [sum(ts.get_lanes_density()) for ts in env.traffic_signals.values()]
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
                "obs": json.dumps(env.get_observations_dict()),
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
        obs = next_obs
    # https://sumo.dlr.de/pydoc/traci._simulation.html#SimulationDomain-getParameter
    return results


def single_episodes(env, agent, prefix):
    results = run_episode(env, agent)
    print(f"./evaluations/{prefix}_1_episode.csv")
    pd.DataFrame(results).to_csv(
        f"./evaluations/{prefix}_1_episode.csv",
        index=False,
    )


def multiple_episodes(env, agent, prefix):
    n_episodes = 100
    results = []

    for episode_number in range(n_episodes):
        print("Episode", episode_number)
        for _ in range(5):
            try:
                episode_results = run_episode(env, agent)
                break
            except Exception as e:
                print("Error in episode", episode_number)
                print(e)
        else:
            raise Exception("Failed to run episode")
        # Episode metrics
        # https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html
        # https://sumo.dlr.de/docs/TraCI/Simulation_Value_Retrieval.html
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
        if n_vehicles == 0:
            avg_travel_time = 0
        else:
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
        f"./evaluations/{prefix}_{n_episodes}_episode.csv",
        index=False,
    )


def main(traffic: str, model: str):
    settings = ROUTE_SETTINGS[traffic]

    route_file = settings["path"]
    start_time = settings["begin_time"]
    end_time = settings["end_time"]
    duration = end_time - start_time
    env = CustomSumoEnvironment(
        net_file=route_file.format(type="net"),
        route_file=route_file.format(type="rou"),
        # single_agent=True,
        begin_time=start_time,
        num_seconds=duration,
    )
    env.reset()
    prefix = f"{model}_{traffic}"

    agent = option_critic_nn.OptionCriticNeuralNetwork(
        env=env,
        num_options=10,
        temperature=0.1,
        eps_start=0.9,
        eps_min=0.1,
        eps_decay=0.999,
        eps_test=0.05,
        device="cpu",
    )
    agent.load_state_dict(
        torch.load(
            "./models/option_critic_nn_10_options_custom-2way-single-intersection_100800_steps"
        )["model_params"]
    )

    single_episodes(env, agent, prefix)
    # multiple_episodes(env, agent, prefix)


if __name__ == "__main__":
    pathlib.Path("./evaluations").mkdir(parents=True, exist_ok=True)
    # parser.add_argument('-m', '--model', required=True)
    # parser.add_argument('-t', '--traffic',
    #                     choices=possible_scenarios,
    #                     required=True)
    # args = parser.parse_args()

    # if args.traffic == 'all':
    #     for scenario in ["cologne1", "cologne3", "cologne8",
    #                      "ingolstadt1", "ingolstadt7", "ingolstadt21",
    #                      "hangzhou_1x1_bc-tyc_18041607_1h", "jinan"]:
    #     # for scenario in ROUTE_SETTINGS.keys():
    #         main(scenario, args.model)
    # else:
    #     main(args.traffic, args.model)

    main("custom-2way-single-intersection", "option_critic_nn")
