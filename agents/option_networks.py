import random

import torch
import torch.nn as nn
import torch.nn.init as init


class ReluNetwork(nn.Module):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )
        self.to(device)
        # self.apply(self.init_weights)


    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            init.constant_(m.weight, -10.0)  # Initialize weights to large negative values
            if m.bias is not None:
                init.constant_(m.bias, -10.0)  # Initialize biases to large negative values


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class MultiDiscreteReluNetwork(nn.Module):
    def __init__(self, input_size, action_sizes, device):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Linear(256, n_actions),
        )
        # self.apply(self.init_weights)
        self.heads = nn.ModuleList(
            nn.Linear(256, int(size)) for size in action_sizes
        )
        self.to(device)


    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            init.constant_(m.weight, -10.0)  # Initialize weights to large negative values
            if m.bias is not None:
                init.constant_(m.bias, -10.0)  # Initialize biases to large negative values


    def forward(self, x):
        x = self.flatten(x)
        items = self.linear_relu_stack(x)
        logits = [head(items) for head in self.heads]
        return logits

class QNetwork(nn.Module):
    def __init__(self, obs_size, action_size, device):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )
        self.to(device)
        # This is required as otherwise it's possible that the random
        # initialization of the weights will cause the network to output
        # a higher Q value for a state/option pair for which we did not train
        # This because the rewards tend to be negative
        # self.apply(self.init_weights)


    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            init.constant_(m.weight, -10.0)  # Initialize weights to large negative values
            if m.bias is not None:
                init.constant_(m.bias, -10.0)  # Initialize biases to large negative values


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class TerminationFunctionNetwork(nn.Module):
    """This neural network predicts the probability of terminating an option.
    """
    def __init__(self, obs_size, n_options, device):
        super().__init__()  
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.Tanh(),
            # nn.Linear(256, 256),
            # nn.LeakyReLU(),
            nn.Linear(256, n_options),
        )
        self.to(device)
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            init.xavier_uniform_(m.weight)  # Xavier initialization
            if m.bias is not None:
                init.constant_(m.bias, random.uniform(0, 1))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits