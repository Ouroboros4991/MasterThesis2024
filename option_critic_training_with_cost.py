# Source: https://github.com/lweitkamp/option-critic-pytorch/blob/master/main.py

import numpy as np
import argparse
import torch
from copy import deepcopy

from agents.option_critic import OptionCriticFeatures
from agents.option_critic_forced import OptionCriticForced
from agents.option_critic_nn import OptionCriticNeuralNetwork

from agents.option_critic_utils import to_tensor
from agents.option_critic_utils import critic_loss as critic_loss_fn
from agents.option_critic_utils import actor_loss as actor_loss_fn

from sumo_rl_environment.custom_env import CustomSumoEnvironment

from utils.experience_replay import ReplayBuffer
from utils.sb3_logger import SB3Logger as Logger
import time


from configs import ROUTE_SETTINGS

TRAFFIC = "custom-2way-single-intersection"
SETTINGS = ROUTE_SETTINGS[TRAFFIC]

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
parser.add_argument("--learning-rate", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount rate")
parser.add_argument(
    "--epsilon-start", type=float, default=1.0, help=("Starting value for epsilon.")
)
parser.add_argument("--epsilon-min", type=float, default=0.1, help="Minimum epsilon.")
parser.add_argument(
    "--epsilon-decay",
    type=float,
    default=50000,
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


def run(args):
    route_file = SETTINGS["path"]
    start_time = SETTINGS["begin_time"]
    end_time = SETTINGS["end_time"]
    duration = end_time - start_time
    experiment_name = (
        f"{args.agent}_{args.num_options}_options_with_deliberation_{TRAFFIC}"
    )
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    option_critic = agents[args.agent](
        in_features=env.observation_space.shape[0] + 1,
        num_actions=env.action_space.n,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=device,
    )
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)

    optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)

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
    while steps < args.max_steps_total:

        cumulative_rewards = 0

        obs, _ = env.reset()
        greedy_option = 0
        current_option = 0
        done = False
        option_termination = False
        episode = 0
        curr_op_len = 1
        obs = np.append(obs, curr_op_len)
        logger.start_episode(steps)
        while not done:
            epsilon = option_critic.epsilon
            if option_termination:
                current_option = (
                    np.random.choice(args.num_options)
                    if np.random.rand() < epsilon
                    else greedy_option
                )
                curr_op_len = 1

            state = option_critic.get_state(to_tensor(obs))
            action, logp, entropy = option_critic.get_action(state, current_option)

            next_obs, reward, done, truncated, info = env.step(action)
            done = done | truncated

            termination_obs = np.append(next_obs, curr_op_len)
            termination_state = option_critic.get_state(to_tensor(termination_obs))
            option_termination, greedy_option = (
                option_critic.predict_option_termination(
                    termination_state, current_option
                )
            )
            if option_termination:
                cost = 1000  # TODO make variable
                reward -= cost * (1 / curr_op_len)
                curr_op_len = 1
                next_obs = np.append(next_obs, curr_op_len)
            else:
                curr_op_len += 1
                next_obs = np.append(next_obs, curr_op_len)

            buffer.push(obs, current_option, reward, next_obs, done)

            actor_loss, critic_loss = None, None
            if len(buffer) > args.batch_size:
                actor_loss = actor_loss_fn(
                    obs,
                    current_option,
                    logp,
                    entropy,
                    reward,
                    done,
                    next_obs,
                    option_critic,
                    option_critic_prime,
                    args,
                )
                loss = actor_loss

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(
                        option_critic, option_critic_prime, data_batch, args
                    )
                    loss += critic_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            # update global steps etc
            steps += 1
            obs = next_obs
            cumulative_rewards += reward
            # average_cumulative_rewards *= 0.95
            # average_cumulative_rewards += 0.05 * cumulative_rewards
        logger.log_episode(
            num_timesteps=steps,
            iteration=episode,
            reward=cumulative_rewards,
        )
        episode += 1
    torch.save(
        {"model_params": option_critic.state_dict()},
        f"models/{experiment_name}_{steps}_steps",
    )


if __name__ == "__main__":

    args = parser.parse_args()
    run(args)
