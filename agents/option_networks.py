import torch
import torch.nn as nn

class ReluNetwork(nn.Module):
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
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_options),
            nn.Softmax(dim=-1),
        )
        self.to(device)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits