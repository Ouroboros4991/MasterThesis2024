# Based on the code found here: https://github.com/lweitkamp/option-critic-pytorch/blob/master/option_critic.py
"""Option critic agent in which the termination forces the agent to choose a new option.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

from math import exp



class OptionCriticForced(nn.Module):
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

        super(OptionCriticForced, self).__init__()

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

        self.features = nn.Sequential(
            nn.Linear(in_features, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU()
        )

        self.Q = nn.Linear(64, num_options)  # Policy-Over-Options
        self.terminations = nn.Linear(64, num_options)  # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 64, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

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
        # next_option = Q.argmax(dim=-1)
        _, indices = torch.topk(
            Q, 2, dim=-1, largest=True, sorted=True
        )  # Returns values and indices
        indices = indices[0]  # Unnest array TODO: check if this causes issues
        next_option = indices[0].item()
        if next_option == current_option:
            next_option = indices[1].item()
        return bool(option_termination.item()), next_option

    def get_terminations(self, state):
        return self.terminations(state).sigmoid()

    def get_action(self, state, option):
        logits = state.data @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

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
