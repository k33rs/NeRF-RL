import numpy as np
import torch
from torch import nn


class ACNetwork(nn.Module):
    def __init__(
            self,
            state_dim, action_dim,
            fc1_args, fc2_args, fc3_args
    ):
        super(ACNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(*fc1_args)
        self.fc2 = nn.Linear(*fc2_args)
        self.fc3 = nn.Linear(*fc3_args)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    @staticmethod
    def init_fanin(size, fanin=None):
        fanin = fanin or size[0]
        v = 1. / np.sqrt(fanin)
        return torch.Tensor(size).uniform_(-v, v)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = self.init_fanin(self.fc1.weight.data.size())
        self.fc2.weight.data = self.init_fanin(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)


class Actor(ACNetwork):
    def __init__(
            self,
            state_dim, action_dim,
            hidden1=400, hidden2=300, init_w=3e-3
    ):
        super(Actor, self).__init__(
            state_dim, action_dim,
            (state_dim, hidden1),
            (hidden1, hidden2),
            (hidden2, action_dim),
        )
        self.init_weights(init_w)

    def forward(self, state):
        out = self.fc1(state)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class Critic(ACNetwork):
    def __init__(
            self,
            state_dim, action_dim,
            hidden1=400, hidden2=300, init_w=3e-3
    ):
        super(Critic, self).__init__(
            state_dim, action_dim,
            (state_dim, hidden1),
            (hidden1 + action_dim, hidden2),
            (hidden2, 1),
        )
        self.init_weights(init_w)

    def forward(self, state, action):
        out = self.fc1(state)
        out = self.relu(out)
        # connect action
        out = torch.cat([out, action], dim=-1)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
