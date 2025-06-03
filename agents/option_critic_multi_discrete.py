# Based on the code found here: https://github.com/lweitkamp/option-critic-pytorch/blob/master/option_critic.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

from math import exp
from agents.option_critic_utils import to_tensor


class OptionCriticMultiDiscrete(nn.Module):
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
        start_min_policy_length=None,
    ):

        super().__init__()

        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test = eps_test
        self.num_steps = 0

        self.in_features = self.calculate_in_features(env)
        self.features_output_dim = 64  # Fixed output dimension for features
        self.features = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, self.features_output_dim),
            nn.ReLU(),
        )

        self.Q = nn.Linear(self.features_output_dim, num_options)  # Policy-Over-Options
        self.terminations = nn.Linear(
            self.features_output_dim, num_options
        )  # Option-Termination

        self.action_dims = [action_dim.n for action_dim in env.action_space]
        self.option_policies = [
            nn.Linear(
                self.features_output_dim, sum(self.action_dims), device=self.device
            )
            for _ in range(num_options)
        ]
        self.to(device)
        self.train(not testing)
        self.start_min_policy_length = start_min_policy_length

        self.reset()

    def prep_state(self, obs):
        """Convert the provided observation to a tensor

        Args:
            obs (Any): Obs provided by the environment
        """
        state = self.features(obs)
        return state

    def calculate_in_features(self, env):
        """Calculate the size of the observation space based on the environment.

        Args:
            env (_type_): _description_

        Returns:
            _type_: _description_
        """
        obs, _ = env.reset()
        converted_obs = self._convert_dict_to_tensor(obs)
        return converted_obs.shape[1]

    def _convert_dict_to_tensor(self, obs):
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

    def get_state(self, obs):
        obs = self._convert_dict_to_tensor(obs)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        q_values = self.Q(state)
        if q_values.ndim == 3:  # Check if the shape is (batch_size, 1, num_options)
            q_values = q_values.squeeze(1)  # Remove the second dimension
        return q_values

    def predict_option_termination(self, state, current_option):
        self.terminations(state)
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()

    def get_terminations(self, state):
        terminations = self.terminations(state).sigmoid()
        if terminations.ndim == 3:  # Check if the shape is (batch_size, 1, num_options)
            terminations = terminations.squeeze(1)  # Remove the second dimension
        return terminations

    def get_action(self, state, fixed_option: int = None):
        # Validate the option
        state = state.to(self.device)

        self.curr_op_len += 1

        new_option = self.current_option
        option_termination, greedy_option = self.predict_option_termination(
            state, self.current_option
        )
        should_terminate = option_termination
        if self.start_min_policy_length:
            if self.curr_op_len < self.start_min_policy_length:
                should_terminate = False
        if should_terminate:
            # density = self.get_lanes_density(self.env)
            # self.option_termination_states[self.current_option].append(density)
            # TODO: make generic
            new_option = (
                np.random.choice(self.num_options)
                if np.random.rand() < self.epsilon
                else greedy_option
            )
        if fixed_option is not None:
            new_option = fixed_option
        if new_option != self.current_option:
            self.update_option_lengths()
            self.curr_op_len = 0
        self.current_option = new_option
        action_logits = self.option_policies[self.current_option](state)
        # Add temperature scaling to encourage exploration
        scaled_logits = action_logits / self.temperature
        # Based on A2C implementation of stablebasaelines
        distributions = [
            Categorical(logits=split)
            for split in torch.split(scaled_logits, list(self.action_dims), dim=1)
        ]
        actions = [dist.sample() for dist in distributions]
        entropy = torch.stack([dist.entropy() for dist in distributions], dim=1).sum(
            dim=1
        )
        log_prob = torch.stack(
            [dist.log_prob(action) for dist, action in zip(distributions, actions)],
            dim=1,
        ).sum(dim=1)
        additional_info = {
            "logp": log_prob.item(),
            "entropy": entropy.item(),
            "termination": option_termination,
            "greedy_option": greedy_option,
        }

        return [action.item() for action in actions], additional_info

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

    def update_option_lengths(self):
        self.option_lengths[self.current_option].append(self.curr_op_len)

    def reset(self):
        self.current_option = 0
        self.curr_op_len = 0
        self.option_lengths = {opt: [] for opt in range(self.num_options)}
