import math
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .config import Config
from ..shared.env import NerfEnv
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
    def __init__(
            self,
            state_dim,
            action_dim,
            config: Config,
            env: NerfEnv,
    ):
        self.actor = Actor(state_dim, action_dim).to(config.device)
        self.critic1 = Critic(state_dim, action_dim).to(config.device)
        self.critic1_target = Critic(state_dim, action_dim).to(config.device)
        self.critic2 = Critic(state_dim, action_dim).to(config.device)
        self.critic2_target = Critic(state_dim, action_dim).to(config.device)

        self.actor_optim = Adam(self.actor.parameters(), lr=config.lr, maximize=True)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=config.lr)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=config.lr)

        self.loss = nn.MSELoss()

        # Make sure target has the same weights
        hard_update(self.critic1_target, self.critic1)
        hard_update(self.critic2_target, self.critic2)

        # replay buffer
        self.memory = SequentialMemory(
            limit=config.replay_bufsize,
            window_length=config.window_length,
        )
        # environment
        self.env = env

        # hyperparameters
        self.batch_size = None # managed by runner
        self.mem_batch_size = config.mem_batch_size
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.tau = config.tau

        self.device = config.device
        self.initial_state = None # initial state
        self.action_dim = action_dim
        self.clip_angle = None # managed by runner
        self.imshape = None # managed by runner
        self.imsize = None # managed by runner
        self.camera_intrinsics = None # managed by runner

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

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
        """sample next action by policy"""
        initial_dir = self.initial_state[..., -self.action_dim:]
        curr_dir = state[..., -self.action_dim:]
        mean = self.actor(state)
        # clip mean to angle
        mean = clip_to_angle(
            dir=mean,
            axis=curr_dir,
            clip_angle_rad=self.clip_angle_small,
        )
        # clip mean to image boundaries
        mean = clip_to_image(
            dir=mean,
            default=initial_dir,
            camera_intrinsics=self.camera_intrinsics,
            imshape=self.imshape,
        )
        # sample actions from Von Mises distribution
        k = 1 / self.clip_angle**2
        action_dist = torch.distributions.VonMises(mean, k)
        next_dir = action_dist.sample((1,)).squeeze(0)
        # compute log probability
        log_c = torch.log(k) - np.log(2* np.pi)
        k_large = k > 500
        log_c[k_large] = log_c[k_large] - k[k_large]
        log_c[~k_large] = log_c[~k_large] - torch.log(torch.exp(k) - torch.exp(-k))[~k_large]
        log_prob = log_c + k * (mean * next_dir).sum(dim=-1, keepdim=True)
        return next_dir, log_prob
    
    def observe(self, state, action, reward, done):
        self.memory.append(
            to_numpy(state),
            to_numpy(action),
            to_numpy(reward),
            done,
        )

    def observe_done(self, state):
        action, _ = self.select_action(state)
        self.memory.append(
            to_numpy(state),
            to_numpy(action),
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

        camera_intrinsics = self.camera_intrinsics.clone() # make a copy
        initial_state = self.initial_state.clone()

        policy_loss_agg = 0
        q1_loss_agg, q2_loss_agg = 0, 0

        for i in range(0, self.batch_size, self.imsize):
            state = to_tensor(state_batch[:, i:i+self.imsize], self.device)
            action = to_tensor(action_batch[:, i:i+self.imsize], self.device)
            next_state = to_tensor(next_state_batch[:, i:i+self.imsize], self.device)
            reward = to_tensor(reward_batch[:, i:i+self.imsize], self.device)
            # index into camera intrinsics (needed by select_action)
            left = i // self.imsize
            right = left + 1
            self.camera_intrinsics = camera_intrinsics[left:right]
            self.initial_state = initial_state[i:i+self.imsize]
            # Critic targets
            with torch.no_grad():
                next_action, next_log_prob = self.select_action(next_state)
                next_q1 = self.critic1_target(next_state, next_action)
                next_q2 = self.critic2_target(next_state, next_action)
                min_q = torch.minimum(next_q1, next_q2)
                next_q = min_q - self.alpha * next_log_prob
                target_q = reward + self.gamma * (1 - done) * next_q
            # Critic update (TD)
            q1 = self.critic1(state, action)
            q2 = self.critic2(state, action)
            q1_loss = self.loss(q1, target_q)
            q2_loss = self.loss(q2, target_q)
            self.critic1.zero_grad()
            self.critic2.zero_grad()
            q1_loss.backward(retain_graph=True)
            q2_loss.backward()
            self.critic1_optim.step()
            self.critic2_optim.step()
            # Actor update
            action_pi, log_prob = self.select_action(state)
            q1_pi = self.critic1(state, action_pi)
            q2_pi = self.critic2(state, action_pi)
            min_q_pi = torch.minimum(q1_pi, q2_pi)
            policy_loss = (min_q_pi - self.alpha * log_prob).mean()
            self.actor.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()
            # Target update
            soft_update(self.critic1_target, self.critic1, self.tau)
            soft_update(self.critic2_target, self.critic2, self.tau)
            # accumulate losses
            q1_loss_agg += q1_loss.item()
            q2_loss_agg += q2_loss.item()
            policy_loss_agg += policy_loss.item()

        self.camera_intrinsics = camera_intrinsics # restore
        self.initial_state = initial_state

        nsteps = math.ceil(self.batch_size / self.imsize)
        policy_loss_agg /= nsteps
        q1_loss_agg /= nsteps
        q2_loss_agg /= nsteps
        return policy_loss_agg, q1_loss_agg, q2_loss_agg


    def load_weights(self, actor_file, critic1_file, critic2_file):
        self.actor.load_state_dict(torch.load(actor_file))
        self.critic1.load_state_dict(torch.load(critic1_file))
        self.critic2.load_state_dict(torch.load(critic2_file))

    def save_model(self, path):
        torch.save(self.actor.state_dict(), f'{path}/actor.pkl')
        torch.save(self.critic1.state_dict(), f'{path}/critic1.pkl')
        torch.save(self.critic2.state_dict(), f'{path}/critic2.pkl')
