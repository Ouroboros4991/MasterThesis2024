# Based on the code found here: https://github.com/lweitkamp/option-critic-pytorch/blob/master/option_critic.py

import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

from math import exp
import numpy as np

from agents.option_critic_utils import to_tensor
from agents.inter_option_policy import InterOptionPolicyNetwork


class OptionCriticNeuralNetwork(nn.Module):
    def __init__(
        self,
        in_features,
        num_actions,
        num_options,
        temperature=1.0,
        eps_start=1.0,
        eps_min=0.1,
        eps_decay=int(1e6),
        eps_test=0.05,
        device="cpu",
        testing=False,
    ):

        super(OptionCriticNeuralNetwork, self).__init__()

        self.in_features = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.num_steps = 0

        # self.features = nn.Sequential(
        #     nn.Linear(in_features, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 64),
        #     nn.ReLU()
        # )

        self.Q = nn.Linear(in_features, num_options)  # Policy-Over-Options
        self.terminations = nn.Linear(in_features, num_options)  # Option-Termination
        self.option_policies = [InterOptionPolicyNetwork(in_features, num_actions, device)
                                for _ in range(num_options)]

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        # state = self.features(obs)
        return obs

    def get_Q(self, state):
        return self.Q(state)

    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()

    def get_terminations(self, state):
        return self.terminations(state).sigmoid()

    def get_action(self, state, option):
        action_dist = self.option_policies[option](state)
        action_dist = Categorical(logits=action_dist)
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy

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