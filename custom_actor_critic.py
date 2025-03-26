# Source: https://github.com/lweitkamp/option-critic-pytorch/blob/master/main.py

import numpy as np
import argparse
import torch
from copy import deepcopy

from agents.actor_critic_agent import CustomActorCritic

from sumo_rl_environment.custom_env import CustomSumoEnvironment

from utils.experience_replay import ReplayBuffer
from utils.experience_replay import OptionRewardReplayBuffer

from utils.sb3_logger import SB3Logger as Logger

import time

from configs import ROUTE_SETTINGS

# TRAFFIC = "custom-2way-single-intersection"


agents = {
    "actor_critic": CustomActorCritic,
}

parser = argparse.ArgumentParser(description="Option Critic PyTorch")

parser.add_argument(
    "--agent",
    type=str,
    default="actor_critic",
    choices=agents.keys(),
    help="Agent to use",
)
parser.add_argument(
    "--optimal-eps", type=float, default=0.05, help="Epsilon when playing optimally"
)
parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
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

parser.add_argument('-t', '--traffic',
                    choices=list(ROUTE_SETTINGS.keys()),
                    required=True) 

def get_lanes_density(env):
    """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

    Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
    """
    lanes_density = [tf.get_lanes_density() for tf in list(env.traffic_signals.values())]
    result = 0
    for item in lanes_density:
        result += sum(item)
    return result


def get_lanes_max_queue(env):
    """
    """
    traffic_lane_dict = {}
    for tf_id, tf in env.traffic_signals.items():
        lane_queues = tf.get_lanes_queue()
        max_index = np.array(lane_queues).argmax()
        traffic_lane_dict[tf_id] = max_index
    return traffic_lane_dict
        

def run(args):
    settings = ROUTE_SETTINGS[args.traffic]

    route_file = settings["path"]
    start_time = settings["begin_time"]
    end_time = settings["end_time"]
    duration = end_time - start_time
    experiment_name = f"{args.agent}_{args.traffic}"
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    actor_critic = agents[args.agent](
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
    actor_critic_prime = deepcopy(actor_critic)

    # optim = torch.optim.RMSprop(actor_critic.parameters(), lr=args.learning_rate)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.learning_rate, weight_decay=0.01)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # env.seed(args.seed)

    buffer = OptionRewardReplayBuffer(capacity=args.max_history, seed=args.seed)
    logger = Logger(
        verbose=3,
        tensorboard_log=args.logdir,
        tb_log_name=f"{experiment_name}-{time.ctime()}",
    )
    
    steps = 0
    action_epsilon = 0.3
    action_epsilon_decay = 0.95
    exploration_episodes = 20
    
    # while steps < args.max_steps_total:
    for episode in range(num_episodes):
        done = False
        steps_since_last_update = 0
        cumulative_rewards = 0

        obs = env.reset()        
        
        logger.start_episode(steps) 
        option_termination_states = {o: [] for o in range(args.num_options)}
        actor_critic.reset()
        option_reward = 0
                        
        while not done:
            state = actor_critic.prep_state(obs)
            action, additional_info = actor_critic.get_action(state)
            # if episode < exploration_episodes or np.random.rand() < action_epsilon:
            #     action = np.random.choice(actor_critic.num_actions)
            logp = additional_info["logp"]
            entropy = additional_info["entropy"]
            action_dict = actor_critic.convert_action_to_dict(action)
            next_obs, rewards, dones, info = env.step(action_dict)
            reward = np.mean(list(rewards.values()))
            next_state = actor_critic.prep_state(next_obs)
            done = dones["__all__"]
            buffer.push(state, actor_critic.current_option, reward, option_reward, next_state, done)
            
            actor_loss, critic_loss = actor_critic.calculate_loss(state,
                                                                  actor_critic.current_option,
                                                                  logp,
                                                                  entropy,
                                                                  reward,
                                                                  done,
                                                                  next_state,
                                                                  args)
            loss = actor_loss + critic_loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            if steps % args.freeze_interval == 0:
                actor_critic_prime.load_state_dict(actor_critic.state_dict())

            # update global steps etc
            steps += 1
            steps_since_last_update += 1
            obs = next_obs
            cumulative_rewards += reward
        actor_critic.update_option_lengths()
        logger.log_episode(
            num_timesteps=steps,
            iteration=episode,
            reward=cumulative_rewards,
            option_lengths={},
        )
        action_epsilon = action_epsilon * action_epsilon_decay
        # min_option_length = min_option_length * args.policy_length_decay
        # print(min_option_length, args.policy_length_decay)
        

    # torch.save(
    #     {"model_params": actor_critic.state_dict()},
    #     f"models/{experiment_name}_{steps}_steps",
    # )
    actor_critic.save(f"models/{experiment_name}_{steps}_steps")


if __name__ == "__main__":
    args = parser.parse_args()
    run(args)
