# Based on the code found here: https://github.com/lweitkamp/option-critic-pytorch/blob/master/option_critic.py

import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
from itertools import permutations

from math import exp
import numpy as np

from agents.option_critic_utils import to_tensor


class OptionCriticMultiTrafficLights(nn.Module):
    """This option critic class is used together with the custom sumo environment
    as redefines the action space so that one agent can manage all traffic lights
    """

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
    ):

        super(OptionCriticMultiTrafficLights, self).__init__()

        self.in_features = env.observation_space.shape[0]
        # Update action space based on the number of traffic lights
        # This is done by getting all possible permutations of the action space and the number of traffic lights
        # The action space contains the amount of possible configurations in 1 traffic light
        # The end result is a list of all possible configurations across the traffic lights
        self.traffic_lights = env.ts_ids
        self.action_space = [
            config
            for config in permutations(
                range(env.action_space.n), len(self.traffic_lights)
            )
        ]
        self.num_actions = len(self.action_space)
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.num_steps = 0

        self.features = nn.Sequential(
            nn.Linear(self.in_features, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU()
        )

        self.Q = nn.Linear(64, num_options)  # Policy-Over-Options
        self.terminations = nn.Linear(64, num_options)  # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 64, self.num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, self.num_actions))

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

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
        logits = state.data @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        # Convert the single action to the correct mapping of configs per traffic light
        action_dict = {}
        for index, a in enumerate(self.action_space[action.item()]):
            traffic_light = self.traffic_lights[index]
            action_dict[traffic_light] = a

        return action_dict, logp, entropy

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
