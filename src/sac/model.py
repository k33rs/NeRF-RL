import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_mean = nn.Linear(hidden, action_dim)
        self.relu = nn.ReLU()

    def forward(self, state):
        out = self.fc1(state)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        mean = self.fc_mean(out)
        return mean


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        out = torch.cat([state, action], dim=-1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
