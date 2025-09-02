
"""
LAB02
Exercise 2

Policy Gradient Method applied to the cartpole environment
"""

# Import external libraries
import torch.nn as nn
import torch
import torch.nn.functional as F

class Policy(nn.Module):
    """
    Policy network for the cartpole environment
    """
    def __init__(self, input_size, actions, softmax=True):
        super(Policy, self).__init__()
        self.softmax = softmax
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, actions)
        )

    def forward(self, x):
        x = self.layers(x)
        if self.softmax:
            x = F.softmax(x, dim=-1)
        return x
    
    def do_action(self, state, device):
        state = torch.from_numpy(state).float.unsqueeze(0).to(device)
        probs = self.forward(state).to(device)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)



class Policy_Lander(nn.Module):
    """
    Policy network for the Lunar Lander environment
    """
    def __init__(self, input_size, actions, softmax=True):
        super(Policy_Lander, self).__init__()
        self.softmax = softmax
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actions)
        )

    def forward(self, x):
        x = self.layers(x)
        if self.softmax:
            x = F.softmax(x, dim=-1)
        return x
    
    def do_action(self, state, device):
        state = torch.from_numpy(state).float.unsqueeze(0).to(device)
        probs = self.forward(state).to(device)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
