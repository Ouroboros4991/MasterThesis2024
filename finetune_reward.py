import argparse
import json
import math
import logging

import numpy as np
import pandas as pd

# import supersuit as ss
import stable_baselines3
import torch

import evaluate
import configs
from sumo_rl_environment.custom_functions import CustomObservationFunction
from utils.utils import setup_env


def train(
    env,
    traffic: str,
    steps: int = 30000,
    reward_fn: str = "pressure",
    broken: bool = False,
):

    if broken:
        experiment_name = f"a2c_broken_{traffic}_{steps}_steps"
    else:
        experiment_name = f"a2c_{traffic}_{steps}_steps"
    experiment_name += "_finetuning"
    experiment_name += reward_fn
    if env.env.reward_weights:
        experiment_name += "_".join(
            [f"{key}_{value}" for key, value in env.env.reward_weights.items()]
        )

    # Check if model already exists:
    try:
        agent = stable_baselines3.A2C.load(f"models/{experiment_name}.zip")
        print(f"Model {experiment_name} already exists, loading...")
        return agent
    except Exception:
        print(f"Model {experiment_name} does not exist, training new model...")
    # env = DummyVecEnv([lambda: env])
    env.reset()

    agent = stable_baselines3.A2C(
        "MultiInputPolicy",
        env,
        verbose=3,
        gamma=0.95,
        tensorboard_log=f"runs/{experiment_name}",
        # learning_rate=0.0001,
        ent_coef=0.02,
    )

    agent.learn(total_timesteps=steps)
    agent.save(f"models/{experiment_name}.zip")
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
        observation = CustomObservationFunction.get_observation_arr(observation_dict)
        state[key] = torch.tensor(
            [observation],
            dtype=torch.float32,
            device=policy_model.device,
        )  # device="cuda")
    return (
        policy_model.get_distribution(state)
        .distribution[0]
        .probs.cpu()
        .detach()
        .numpy()[0]
    )


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


def eval_episode(experiment_name: str, eval_env1, eval_env2, agent1, agent2):
    single_episode_result = evaluate.single_episodes(
        env=eval_env1, agent=agent1, prefix=f"{experiment_name}_env1_agent1", save=False
    )
    hd_arr = [
        hellinger_distance(json.loads(item["obs"]), agent1, agent2)
        for item in single_episode_result
    ]

    reward_arr = [item["cumulative_reward"] for item in single_episode_result]
    avg_hd_distance_env1_agent1 = np.mean(hd_arr)
    avg_reward_env1_agent1 = np.mean(reward_arr)

    single_episode_result = evaluate.single_episodes(
        env=eval_env1, agent=agent2, prefix=f"{experiment_name}_env1_agent2", save=False
    )
    hd_arr = [
        hellinger_distance(json.loads(item["obs"]), agent1, agent2)
        for item in single_episode_result
    ]

    reward_arr = [item["cumulative_reward"] for item in single_episode_result]
    avg_hd_distance_env1_agent2 = np.mean(hd_arr)
    avg_reward_env1_agent2 = np.mean(reward_arr)

    single_episode_result = evaluate.single_episodes(
        env=eval_env2, agent=agent1, prefix=f"{experiment_name}_env2_agent1", save=False
    )
    hd_arr = [
        hellinger_distance(json.loads(item["obs"]), agent1, agent2)
        for item in single_episode_result
    ]

    reward_arr = [item["cumulative_reward"] for item in single_episode_result]
    avg_hd_distance_env2_agent1 = np.mean(hd_arr)
    avg_reward_env2_agent1 = np.mean(reward_arr)

    single_episode_result = evaluate.single_episodes(
        env=eval_env2, agent=agent2, prefix=f"{experiment_name}_env2_agent2", save=False
    )
    hd_arr = [
        hellinger_distance(json.loads(item["obs"]), agent1, agent2)
        for item in single_episode_result
    ]

    reward_arr = [item["cumulative_reward"] for item in single_episode_result]
    avg_hd_distance_env2_agent2 = np.mean(hd_arr)
    avg_reward_env2_agent2 = np.mean(reward_arr)
    return {
        "avg_hd_distance_env1_agent1": avg_hd_distance_env1_agent1,
        "avg_reward_env1_agent1": avg_reward_env1_agent1,
        "avg_hd_distance_env1_agent2": avg_hd_distance_env1_agent2,
        "avg_reward_env1_agent2": avg_reward_env1_agent2,
        "avg_hd_distance_env2_agent1": avg_hd_distance_env2_agent1,
        "avg_reward_env2_agent1": avg_reward_env2_agent1,
        "avg_hd_distance_env2_agent2": avg_hd_distance_env2_agent2,
        "avg_reward_env2_agent2": avg_reward_env2_agent2,
    }


def generate_possible_intelli_light_reward_weights():
    reward_weight_values = {
        # 'queue': [0,1,2,3],
        "delay": [0, 1, 2, 3],
        "waiting_time": [0, 1, 2, 3, 10],
        "light_switches": [0, 1, 2, 3],
    }
    reward_configs = []
    # for queue_weight in reward_weight_values['queue']:
    for delay_weight in reward_weight_values["delay"]:
        for waiting_time_weight in reward_weight_values["waiting_time"]:
            for light_switches_weight in reward_weight_values["light_switches"]:
                reward_configs.append(
                    {
                        # 'queue': queue_weight,
                        "delay": delay_weight,
                        "waiting_time": waiting_time_weight,
                        "light_switches": light_switches_weight,
                    }
                )
    return reward_configs


def main(reward_fn: str):
    experiment_prefix = f"finetuning_{reward_fn}"
    try:
        already_run = pd.read_csv(f"./evaluations/{experiment_prefix}.csv")
        known_weights = [json.loads(item) for item in already_run.config.to_list()]
        final_results = already_run.to_dict(orient="records")
    except FileNotFoundError:
        known_weights = []
        final_results = []
    print(known_weights)
    if reward_fn == "intelli_light_reward":
        traffic1 = "custom-2way-single-intersection-low"
        traffic2 = "custom-2way-single-intersection-high"
        broken1 = False
        broken2 = False
        reward_weights = generate_possible_intelli_light_reward_weights()
    elif reward_fn == "intelli_light_prcol_reward":
        traffic1 = "3x3grid-3lanes2"
        traffic2 = "3x3grid-3lanes2"
        broken1 = False
        broken2 = True
        out_lanes_availability_weights = [0, 1, 2, 3, 5, 10]
        reward_weights = []
        for out_lanes_availability in out_lanes_availability_weights:
            base_reward_weights = configs.INTELLI_LIGHT_REWARD.copy()
            base_reward_weights["out_lanes_availability"] = out_lanes_availability
            reward_weights.append(base_reward_weights)

    else:
        raise ValueError(f"Unknown reward function: {reward_fn}")
    for weights in reward_weights:
        if weights in known_weights:
            print(f"Already run {weights}")
            continue
        print(f"TESTING FOR {weights}")
        for tries in range(3):
            try:
                env1 = setup_env(
                    traffic1, reward_fn, weights, broken=broken1, target_model="a2c"
                )
                env2 = setup_env(
                    traffic2, reward_fn, weights, broken=broken2, target_model="a2c"
                )
            except Exception as e:
                print("Error in creating env:", e)
                print(f"Retrying {tries+1}...")
        training_steps = 50000
        for tries in range(3):
            try:
                agent1 = train(
                    env1,
                    traffic1,
                    training_steps,
                    reward_fn,
                    broken=broken1,
                )
                agent2 = train(
                    env2,
                    traffic2,
                    training_steps,
                    reward_fn,
                    broken=broken2,
                )

                cross_episode_results = []
                for eval_tries in range(3):
                    try:
                        eval_env1 = setup_env(
                            traffic1,
                            "pressure",
                            reward_weights=weights,
                            broken=broken1,
                            target_model="a2c",
                        )
                        eval_env2 = setup_env(
                            traffic2,
                            "pressure",
                            reward_weights=weights,
                            broken=broken2,
                            target_model="a2c",
                        )
                    except Exception as e:
                        print("Error in creating eval env:", e)
                        print(f"Retrying {tries+1}...")
                    for _ in range(10):
                        try:
                            result_dict = eval_episode(
                                experiment_prefix, eval_env1, eval_env2, agent1, agent2
                            )
                            cross_episode_results.append(result_dict)
                        except Exception:
                            result_dict = eval_episode(
                                experiment_prefix, eval_env1, eval_env2, agent1, agent2
                            )
                            cross_episode_results.append(result_dict)
                    avg_result = {
                        "config": json.dumps(weights),
                        "avg_hd_distance_env1_agent1": np.mean(
                            [
                                item["avg_hd_distance_env1_agent1"]
                                for item in cross_episode_results
                            ]
                        ),
                        "avg_reward_env1_agent1": np.mean(
                            [
                                item["avg_reward_env1_agent1"]
                                for item in cross_episode_results
                            ]
                        ),
                        "avg_hd_distance_env1_agent2": np.mean(
                            [
                                item["avg_hd_distance_env1_agent2"]
                                for item in cross_episode_results
                            ]
                        ),
                        "avg_reward_env1_agent2": np.mean(
                            [
                                item["avg_reward_env1_agent2"]
                                for item in cross_episode_results
                            ]
                        ),
                        "avg_hd_distance_env2_agent1": np.mean(
                            [
                                item["avg_hd_distance_env2_agent1"]
                                for item in cross_episode_results
                            ]
                        ),
                        "avg_reward_env2_agent1": np.mean(
                            [
                                item["avg_reward_env2_agent1"]
                                for item in cross_episode_results
                            ]
                        ),
                        "avg_hd_distance_env2_agent2": np.mean(
                            [
                                item["avg_hd_distance_env2_agent2"]
                                for item in cross_episode_results
                            ]
                        ),
                        "avg_reward_env2_agent2": np.mean(
                            [
                                item["avg_reward_env2_agent2"]
                                for item in cross_episode_results
                            ]
                        ),
                    }
                    final_results.append(avg_result)
                    print(avg_result)
                    print("=========================")

                    pd.DataFrame(final_results).to_csv(
                        f"./evaluations/{experiment_prefix}.csv",
                        index=False,
                    )
                    break
                break
            except Exception as e:
                print("Error in training or evaluation:")
                logging.error(e)
                print(f"Retrying {tries+1}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--reward_fn",
        type=str,
        help="Reward function to fine-tune",
    )
    args = parser.parse_args()
    main(args.reward_fn)
