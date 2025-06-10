# Based on the code found here: https://github.com/lweitkamp/option-critic-pytorch/blob/master/option_critic.py

import itertools
import joblib
import torch
import torch.nn as nn
from torch.distributions import Categorical

from math import exp

from agents.option_critic_utils import to_tensor
from agents.option_networks import ReluNetwork
from agents.option_networks import TerminationFunctionNetwork
from agents.option_networks import QNetwork


class OptionCriticClassification(nn.Module):
    def __init__(
        self,
        env,
        num_options,
        temperature=1.0,
        eps_start=1.0,
        eps_min=0.1,
        eps_decay=int(1e6),
        eps_test=0.05,
        device="cpu",
        testing=False,
        start_min_policy_length=None        
    ):

        super(OptionCriticClassification, self).__init__()
        
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.num_steps = 0
        self.current_option = 0
        
        self.num_options = num_options
        # self.option_termination_states = {o: [] for o in range(self.num_options)}
        
        # self.env = env
        self.init_actions(env)
        
        self.num_actions = len(self.action_list)
        self.in_features = self.calculate_in_features(env)
        
        # self.features = nn.Sequential(
        #     nn.Linear(in_features, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 64),
        #     nn.ReLU()
        # )

        # self.Q = nn.Linear(self.in_features, num_options)  # Policy-Over-Options
        self.Q = QNetwork(self.in_features, num_options, device)
        self.terminations = TerminationFunctionNetwork(self.in_features, self.num_options, device)
        # self.terminations = nn.Linear(in_features, num_options)  # Option-Termination
        self.option_policies = [ReluNetwork(self.in_features, self.num_actions, device)
                                for _ in range(num_options)]

        self.to(device)
        self.train(not testing)
        self.start_min_policy_length = start_min_policy_length
        
        self.reset()
        
        # TODO: make dynamic
        self.option_classification = joblib.load("./models/option_classifier.joblib")
        self.critic_network = ReluNetwork(self.in_features, 1, device)
    
    def prep_state(self, obs):
        """Convert the provided observation to a tensor

        Args:
            obs (Any): Obs provided by the environment
        """
        
        # Unnest observation
        obs_array = []
        if isinstance(obs, dict):
            for _, obs_arr in obs.items():
                obs_array.extend(obs_arr)
        else:
            obs_array.extend(obs)
        
        
        # include option information
        # Simplified to make it easier to manage
        # encoded_option = np.zeros(self.num_options)
        # encoded_option[self.current_option] = 1
        # obs_array = np.append(obs_array, encoded_option)
        
        # Convert to tensor
        obs_tensor = to_tensor(obs_array)
        if obs_tensor.ndim < 4:
            obs_tensor = obs_tensor.unsqueeze(0)
        obs_tensor = obs_tensor.to(self.device)
        return obs_tensor
        
        
    def calculate_in_features(self, env):
        """Calculate the size of the observation space based on the environment.

        Args:
            env (_type_): _description_

        Returns:
            _type_: _description_
        """        
        obs = env.reset()
        converted_obs = self.prep_state(obs)
        return converted_obs.shape[1]
    
    def init_actions(self, env):
        self.actions_per_traffic_light = {}
        for tf_id, traffic_light in env.traffic_signals.items():
            self.actions_per_traffic_light[tf_id] = [i for i in range(traffic_light.num_green_phases)]
        possible_actions = list(self.actions_per_traffic_light.values())
        if len(possible_actions) == 1:
            self.action_list = possible_actions[0]
        else:
            self.action_list = list(itertools.product(*possible_actions))

    
    def convert_action_to_dict(self, action: int) -> dict:
        """Convert the action from integer back to a dict
        """
        action_tuple = self.action_list[action]
        action_dict = {}
        index = 0
        for tf, possible_actions in self.actions_per_traffic_light.items():
            if isinstance(action_tuple, int):
                if index > 0:
                    raise Exception("Expected action to be a tuple when using multiple traffic lights")
                tf_action = action_tuple
            else:
                tf_action = action_tuple[index]
                if tf_action not in possible_actions:
                    print(tf, tf_action, possible_actions)
                    raise Exception("Invalid action")
            action_dict[tf] = tf_action
            index += 1
        return action_dict


    def get_Q(self, state):
        return self.Q(state)

    def predict_option_termination(self, state, current_option):
        next_option = self.option_classification.predict(state.cpu().detach().numpy())[0]
        option_termination = next_option != current_option
        return option_termination, next_option
    
    def get_terminations(self, state):
        # TODO: cleanup
        terminations = self.terminations(state)
        terminations = terminations.sigmoid()
        return terminations

    # def get_lanes_density(env):
    #     """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection.

    #     Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
    #     """
    #     lanes_density = [tf.get_lanes_density() for tf in list(env.traffic_signals.values())]
    #     result = 0
    #     for item in lanes_density:
    #         result += sum(item)
    #     return result

    def get_action(self, state, fixed_option: int = None):
        # Validate the option
        self.curr_op_len += 1
        
        new_option = self.current_option
        option_termination, greedy_option = (
            self.predict_option_termination(state, self.current_option)
        )
        should_terminate = option_termination
        
        # Min policy length is required to avoid switching to much when we're in states that are between "cluster"
        if self.start_min_policy_length:
            if self.curr_op_len < self.start_min_policy_length:
                should_terminate = False
        if should_terminate:
            new_option = greedy_option
        else:
            new_option = self.current_option
        if new_option != self.current_option:
            self.update_option_lengths()
            self.curr_op_len = 0
        self.current_option = new_option
        action_dist = self.option_policies[self.current_option](state)

        action_dist = Categorical(logits=action_dist)
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        additional_info = {
            "logp": logp,
            "entropy": entropy,
            "termination": option_termination,
            "greedy_option": greedy_option,
        }
        return action.item(), additional_info

    def update_option_lengths(self):
        self.option_lengths[self.current_option].append(self.curr_op_len )
    
    
    def reset(self):
        self.current_option = 0
        self.curr_op_len = 0
        self.option_lengths = {opt: [] for opt in range(self.num_options)}

    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(
                -self.num_steps / self.eps_decay
            )
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps
    
    
    def save(self, path: str):
        state_dict = {
            "Q": self.Q.state_dict(),
            "terminations": self.terminations.state_dict(),
        }
        for i, policy in enumerate(self.option_policies):
            state_dict[f"option_policy_{i}"] = policy.state_dict()
        torch.save(state_dict, path)
        
    def load(self, path: str, map_location: str="cpu"):
        state_dict = torch.load(path, map_location="cpu")
        self.Q.load_state_dict(state_dict["Q"])
        self.terminations.load_state_dict(state_dict["terminations"])
        for i, policy in enumerate(self.option_policies):
            policy.load_state_dict(state_dict[f"option_policy_{i}"])
    
    
    def calculate_loss(self, obs, option, logp, entropy, reward, done, next_obs,  args):
        # Based on https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
        # https://github.com/chengxi600/RLStuff/blob/master/Actor-Critic/Actor-Critic_TD_0.ipynb
        state = obs
        value = self.critic_network(state)
        
        next_state = next_obs
        next_value = self.critic_network(next_state)
        
        mse_loss = nn.MSELoss()
        critic_loss = mse_loss(value, reward + (args.gamma * next_value))
        
        advantage = reward + (args.gamma * next_value) - value
        actor_loss = -logp * advantage
        return actor_loss, critic_loss