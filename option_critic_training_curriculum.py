# Source: https://github.com/lweitkamp/option-critic-pytorch/blob/master/main.py

import numpy as np
import argparse
import torch
from copy import deepcopy

from agents.option_critic import OptionCriticFeatures
from agents.option_critic_forced import OptionCriticForced
from agents.option_critic_nn import OptionCriticNeuralNetwork

from agents.option_critic_utils import critic_loss as critic_loss_fn
from agents.option_critic_utils import actor_loss as actor_loss_fn

from sumo_rl_environment.custom_env import CustomSumoEnvironment

from utils.experience_replay import ReplayBuffer

from utils.sb3_logger import SB3Logger as Logger

import time

from configs import ROUTE_SETTINGS


def get_option(options_dict: dict, current_step: int):
    for key, value in options_dict.items():
        start, end = key.split("-")
        if int(start) <= current_step <= int(end):
            return value
    return 0

# TRAFFIC = "custom-2way-single-intersection"
TRAFFIC = "custom-2way-single-intersection3"
SETTINGS = ROUTE_SETTINGS[TRAFFIC]

OPTIONS_DICT = {
    "0-1000": 0,
    "1000-2000": 1,
    "2000-3000": 0,
    "3000-4000": 2,
    "4000-5000": 0,
}

agents = {
    "option_critic": OptionCriticFeatures,
    "option_critic_forced": OptionCriticForced,
    "option_critic_nn": OptionCriticNeuralNetwork,
}

parser = argparse.ArgumentParser(description="Option Critic PyTorch")

parser.add_argument(
    "--agent",
    type=str,
    default="option_critic",
    choices=agents.keys(),
    help="Agent to use",
)
parser.add_argument(
    "--optimal-eps", type=float, default=0.05, help="Epsilon when playing optimally"
)
parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount rate")
parser.add_argument(
    "--epsilon-start", type=float, default=1.0, help=("Starting value for epsilon.")
)
parser.add_argument("--epsilon-min", type=float, default=0.1, help="Minimum epsilon.")
parser.add_argument(
    "--epsilon-decay",
    type=float,
    default=30000,
    help=("Number of steps to minimum epsilon."),
)
parser.add_argument(
    "--max-history",
    type=int,
    default=10000,
    help=("Maximum number of steps stored in replay"),
)
parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
parser.add_argument(
    "--freeze-interval",
    type=int,
    default=200,
    help=("Interval between target freezes."),
)
parser.add_argument(
    "--update-frequency",
    type=int,
    default=4,
    help=("Number of actions before each SGD update."),
)
parser.add_argument(
    "--termination-reg",
    type=float,
    default=0.01,
    help=("Regularization to decrease termination prob."),
)
parser.add_argument(
    "--entropy-reg",
    type=float,
    default=0.01,
    help=("Regularization to increase policy entropy."),
)
parser.add_argument(
    "--num_options", type=int, default=2, help=("Number of options to create.")
)
parser.add_argument(
    "--temp",
    type=float,
    default=1,
    help="Action distribution softmax tempurature param.",
)

parser.add_argument(
    "--max_steps_total",
    type=int,
    default=int(4e6),
    help="number of maximum steps to take.",
)  # bout 4 million
parser.add_argument(
    "--cuda",
    type=bool,
    default=True,
    help="Enable CUDA training (recommended if possible).",
)
parser.add_argument(
    "--seed", type=int, default=0, help="Random seed for numpy, torch, random."
)
parser.add_argument(
    "--logdir", type=str, default="runs", help="Directory for logging statistics"
)

parser.add_argument(
    "--hd_reg", action="store_true", help="Apply Hellinger Distance Regularization"
)

parser.add_argument(
    "--start_min_policy_length", type=int, default=4, help="Ensure that the option policy runs for at least n steps when starting the training"
)

parser.add_argument(
    "--fixed_learning", type=int, default=100000, help="Number of steps to run with fixed options"
)


def get_lanes_density(env):
    """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

    Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
    """
    lanes_density = [tf.get_lanes_density() for tf in list(env.traffic_signals.values())]
    result = 0
    for item in lanes_density:
        result += sum(item)
    return result


def run(args):
    route_file = SETTINGS["path"]
    start_time = SETTINGS["begin_time"]
    end_time = SETTINGS["end_time"]
    duration = end_time - start_time
    experiment_name = f"{args.agent}_{args.fixed_learning}_steps_curriculum_{args.num_options}_options_{TRAFFIC}"
    if args.hd_reg:
        experiment_name += "_hd_reg"
    # delta_time (int) – Simulation seconds between actions. Default: 5 seconds
    env = CustomSumoEnvironment(
        net_file=route_file.format(type="net"),
        route_file=route_file.format(type="rou"),
        # single_agent=True,
        begin_time=start_time,
        num_seconds=duration,
    )
    env.reset()

    num_episodes = int(args.max_steps_total / duration) * 5  # number of steps simulated between each time period we can take an action
    fixed_episode = int(args.fixed_learning / duration) * 5

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    option_critic = agents[args.agent](
        env=env,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=device,
        start_min_policy_length=args.start_min_policy_length,
        
    )
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)

    # optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)
    optim = torch.optim.Adam(option_critic.parameters(), lr=args.learning_rate, weight_decay=0.01)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # env.seed(args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    logger = Logger(
        verbose=3,
        tensorboard_log=args.logdir,
        tb_log_name=f"{experiment_name}-{time.ctime()}",
    )
    
    steps = 0
    # while steps < args.max_steps_total:
    for episode in range(num_episodes):
        done = False
        steps_since_last_update = 0
        cumulative_rewards = 0

        obs = env.reset()        
        
        logger.start_episode(steps) 
        option_termination_states = {o: [] for o in range(args.num_options)}
        option_critic.reset()

        previous_phases = {tf_id: 0 for tf_id in env.traffic_signals.keys()}

        while not done:
            if episode < fixed_episode:
                fixed_option = get_option(OPTIONS_DICT, env.sim_step)
            else:
                fixed_option = None
            current_option = option_critic.current_option
            density = get_lanes_density(env)


            state = option_critic.prep_state(obs)
            action, additional_info = option_critic.get_action(state, fixed_option)
            logp = additional_info["logp"]
            entropy = additional_info["entropy"]
            if additional_info["termination"]:
                option_termination_states[current_option].append(density)
            action_dict = option_critic.convert_action_to_dict(action)
            next_obs, rewards, dones, info = env.step(action_dict)
            reward = np.mean(list(rewards.values()))
            
            next_state = option_critic.prep_state(next_obs)
            done = dones["__all__"]
            buffer.push(state, option_critic.current_option, reward, next_state, done)
            

            actor_loss, critic_loss = None, None
            if len(buffer) > args.batch_size:
                actor_loss = actor_loss_fn(
                    state,
                    option_critic.current_option,
                    logp,
                    entropy,
                    reward,
                    done,
                    next_state,
                    option_critic,
                    option_critic_prime,
                    args,
                    option_densities=option_termination_states
                )
                loss = actor_loss

                if steps_since_last_update >= args.update_frequency:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(
                        option_critic, option_critic_prime, data_batch, args
                    )
                    loss += critic_loss
                    steps_since_last_update = 0
                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            # update global steps etc
            steps += 1
            steps_since_last_update += 1
            obs = next_obs
            cumulative_rewards += reward
        option_critic.update_option_lengths()
        logger.log_episode(
            num_timesteps=steps,
            iteration=episode,
            reward=cumulative_rewards,
            option_lengths=option_critic.option_lengths
        )
        # min_option_length = min_option_length * args.policy_length_decay
        # print(min_option_length, args.policy_length_decay)
        

    # torch.save(
    #     {"model_params": option_critic.state_dict()},
    #     f"models/{experiment_name}_{steps}_steps",
    # )
    option_critic.save(f"models/{experiment_name}_{steps}_steps")


if __name__ == "__main__":
    args = parser.parse_args()
    run(args)
