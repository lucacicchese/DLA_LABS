import torch.nn as nn
import torch

class Policy(nn.Module):
    def __init__(self, input_size, actions, softmax=True):
        super(Policy, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, actions),
            nn.Softmax(dim=-1) if softmax else nn.Identity()
        )

    def forward(self, x):
        return self.layers(x)
    
    def do_action(self, state, device):
        state = torch.from_numpy(state).float.unsqueeze(0).to(device)
        probs = self.forward(state).to(device)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
