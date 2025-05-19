# Based on the code found here: https://github.com/lweitkamp/option-critic-pytorch/blob/master/option_critic.py

import itertools
import torch
import torch.nn as nn
from torch.distributions import Categorical

from math import exp

from agents.option_critic_utils import to_tensor
from agents.option_networks import ReluNetwork


class CustomActorCritic(nn.Module):
    def __init__(
        self,
        env,
        temperature=1.0,
        eps_start=1.0,
        eps_min=0.1,
        eps_decay=int(1e6),
        eps_test=0.05,
        device="cpu",
        testing=False,
        *args,
        **kwargs
    ):

        super(CustomActorCritic, self).__init__()
        
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.num_steps = 0
        self.current_option = 0
        
        
        # self.env = env
        self.init_actions(env)
        
        self.num_actions = len(self.action_list)
        self.in_features = self.calculate_in_features(env)
    
        self.actor_policy = ReluNetwork(self.in_features, self.num_actions, device)
        self.critic_network = ReluNetwork(self.in_features, 1, device)

        self.to(device)
        self.train(not testing)
        
        self.reset()
        
    
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


    def get_action(self, state, fixed_option: int = None):
        # Validate the option
        action_dist = self.actor_policy(state)
        # print(action_dist)
        action_dist = Categorical(logits=action_dist)
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        additional_info = {
            "logp": logp,
            "entropy": entropy,
            "termination": False,
            "greedy_option": 0,
        }
        # print(action)
        return action.item(), additional_info

    def update_option_lengths(self):
        pass
    
    def reset(self):
        self.current_option = 0
        self.curr_op_len = 0

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
            "actor": self.actor_policy.state_dict(),
            "critic": self.critic_network.state_dict(),
        }
        torch.save(state_dict, path)
        
    def load(self, path: str, map_location: str="cpu"):
        state_dict = torch.load(path, map_location="cpu")
        self.actor_policy.load_state_dict(state_dict["actor"])
        self.critic_network.load_state_dict(state_dict["critic"])
    
    
    def calculate_loss(self, obs, option, logp, entropy, reward, done, next_obs,  gamma):
        # Based on https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py
        # https://github.com/chengxi600/RLStuff/blob/master/Actor-Critic/Actor-Critic_TD_0.ipynb
        state = obs
        value = self.critic_network(state)
        
        next_state = next_obs
        next_value = self.critic_network(next_state)
        
        mse_loss = nn.MSELoss()
        critic_loss = mse_loss(value, reward + (gamma * next_value))
        
        advantage = reward + (gamma * next_value) - value
        actor_loss = -logp * advantage
        return actor_loss, critic_loss