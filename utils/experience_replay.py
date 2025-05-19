# Source: https://github.com/lweitkamp/option-critic-pytorch/blob/master/experience_replay.py

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, capacity, seed=42):
        self.rng = random.SystemRandom(seed)
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, option, reward, next_obs, done):
        self.buffer.append((obs, option, reward, next_obs, done))

    def sample(self, batch_size):
        obs, option, reward, next_obs, done = zip(
            *self.rng.sample(self.buffer, batch_size)
        )
        return torch.stack(obs), np.array(option), np.array(reward), torch.stack(next_obs), np.array(done)

    def __len__(self):
        return len(self.buffer)



class OptionRewardReplayBuffer(object):
    def __init__(self, capacity, seed=42):
        self.rng = random.SystemRandom(seed)
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, option, reward, option_reward, next_obs, done):
        self.buffer.append((obs, option, reward, option_reward, next_obs, done))

    def sample(self, batch_size):
        obs, option, reward, option_reward, next_obs, done = zip(
            *self.rng.sample(self.buffer, batch_size)
        )
        return torch.stack(obs), np.array(option), np.array(reward), np.array(option_reward), torch.stack(next_obs), np.array(done)

    def __len__(self):
        return len(self.buffer)
