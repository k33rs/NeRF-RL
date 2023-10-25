import math
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .config import Config
from ..shared.geom import (
    clip_to_angle,
    clip_to_image,
    random_dir,
)
from .model import Actor, Critic
from ..shared.memory import SequentialMemory
from ..shared.utils import (
    hard_update,
    soft_update,
    to_tensor,
    to_numpy,
)


class Agent:
    def __init__(self, state_dim, action_dim, config: Config):
        self.actor = Actor(state_dim, action_dim, **config.net).to(config.device)
        self.actor_target = Actor(state_dim, action_dim, **config.net).to(config.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=config.actor_lr, maximize=True)

        self.critic = Critic(state_dim, action_dim, **config.net).to(config.device)
        self.critic_target = Critic(state_dim, action_dim, **config.net).to(config.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=config.critic_lr)

        self.loss = nn.MSELoss()

        # Make sure target has the same weights
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # replay buffer
        self.memory = SequentialMemory(
            limit=config.replay_bufsize,
            window_length=config.window_length,
        )

        # hyperparameters
        self.batch_size = None # managed by runner
        self.mem_batch_size = config.mem_batch_size
        self.tau = config.tau
        self.gamma = config.gamma
        self.eps = config.eps
        self.eps_decay = config.eps_decay

        self.device = config.device
        self.initial_state = None # initial state
        self.action_dim = action_dim
        self.clip_angle = None # managed by runner
        self.clip_angle_small = None # managed by runner
        self.imshape = None # managed by runner
        self.imsize = None # managed by runner
        self.camera_intrinsics = None # managed by runner
        self.is_training = True

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def random_action(self, _):
        """random move"""
        initial_dir = self.initial_state[..., -self.action_dim:]
        # perturb within clip angle
        next_dir = random_dir(initial_dir, self.clip_angle)
        # clip to image boundaries
        next_dir = clip_to_image(
            next_dir,
            initial_dir,
            self.camera_intrinsics,
            self.imshape,
        )
        return next_dir

    def select_action(self, state):
        """select next action by policy and clip it"""
        initial_dir = self.initial_state[..., -self.action_dim:]
        curr_dir = state[..., -self.action_dim:]
        next_dir = self.actor(state)
        # add Gaussian noise for exploration
        if self.is_training:
            next_dir = random_dir(
                dir=next_dir,
                angle_rad=max(self.eps, 0) * self.clip_angle_small,
                angle_sample='normal',
            )
            # decay epsilon
            self.eps -= self.eps_decay
        # clip w.r.t. old direction
        next_dir = clip_to_angle(
            dir=next_dir,
            axis=curr_dir,
            clip_angle_rad=self.clip_angle_small,
        )
        # clip to image boundaries
        next_dir = clip_to_image(
            dir=next_dir,
            default=initial_dir,
            camera_intrinsics=self.camera_intrinsics,
            imshape=self.imshape,
        )

        return next_dir
    
    def observe(self, state, action, reward, done):
        self.memory.append(
            to_numpy(state),
            to_numpy(action),
            to_numpy(reward),
            done,
        )

    def observe_done(self, state):
        self.memory.append(
            to_numpy(state),
            to_numpy(self.select_action(state)),
            np.zeros((self.batch_size, 1)),
            False,
        )

    def forget(self):
        self.memory.reset()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch \
            = self.memory.sample_and_split(self.mem_batch_size)
        done = to_tensor(done_batch, self.device).unsqueeze(-1).unsqueeze(-1)

        policy_loss_agg = 0
        value_loss_agg = 0

        for i in range(0, self.batch_size, self.imsize):
            state = to_tensor(state_batch[:, i:i+self.imsize], self.device)
            action = to_tensor(action_batch[:, i:i+self.imsize], self.device)
            next_state = to_tensor(next_state_batch[:, i:i+self.imsize], self.device)
            reward = to_tensor(reward_batch[:, i:i+self.imsize], self.device)
            # Critic update (TD)
            target_action = self.actor_target(next_state)
            next_q_values = self.critic_target(next_state, target_action)
            q_batch = self.critic(state, action)
            target_q_batch = reward + self.gamma * (1 - done) * next_q_values
            value_loss = self.loss(q_batch, target_q_batch)
            self.critic.zero_grad()
            value_loss.backward()
            self.critic_optim.step()
            # Actor update
            next_action = self.actor(state)
            policy_loss = self.critic(state, next_action).mean()
            self.actor.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()
            # Target update
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
            # accumulate losses
            value_loss_agg += value_loss.item()
            policy_loss_agg += policy_loss.item()

        nsteps = math.ceil(self.batch_size / self.imsize)
        policy_loss_agg /= nsteps
        value_loss_agg /= nsteps
        return policy_loss_agg, value_loss_agg


    def load_weights(self, actor_file, critic_file):
        self.actor.load_state_dict(torch.load(actor_file))
        self.critic.load_state_dict(torch.load(critic_file))

    def save_model(self, path):
        torch.save(self.actor.state_dict(), f'{path}/actor.pkl')
        torch.save(self.critic.state_dict(), f'{path}/critic.pkl')
