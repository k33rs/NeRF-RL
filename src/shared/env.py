import gym
import numpy as np
import torch
from gym import spaces
from copy import deepcopy
from typing import TypeVar
from .geom import camera_to_world


RayBundle = TypeVar('RayBundle')


class NerfEnv(gym.Env):
    def __init__(
            self,
            model,
            rad_thres: float=1.,
            rad_max: float=1.,
            reward_scale: float=1.,
            reward_max_resolution: int=1,
            with_radiance=False
    ):
        super(NerfEnv, self).__init__()
        # Initialize environment
        self.rb = None # RayBundle
        self.initial_directions = None
        self.rad_thres = rad_thres
        self.rad_max = rad_max
        self.reward_scale = reward_scale
        self.reward_max_resolution = reward_max_resolution
        self.with_penalty = False # managed by runner
        self.with_radiance = with_radiance
        self.model = model
        self.observation_space = spaces.Box(
            low=np.array([-np.inf for _ in range(6)]),
            high=np.array([np.inf for _ in range(6)]),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-np.inf for _ in range(3)]),
            high=np.array([np.inf for _ in range(3)]),
            dtype=np.float32,
        )
    
    @property
    def directions(self):
        return self.rb.directions
    
    @property
    def imsize(self):
        return self.rb.imsize

    @property
    def px_dist(self):
        return self.rb.px_dist
    
    @property
    def reward_max(self):
        reward = ((self.rad_max - self.rad_thres) / self.rad_max) * self.reward_scale
        return torch.tensor(reward, device=self.rb.device)

    @property
    def state(self):
        if self.rb is None:
            raise RuntimeError('state must be initialized')
        return torch.cat((
            self.rb.origins.view(-1, self.rb.origins.size(-1)),
            self.rb.directions.view(-1, self.rb.directions.size(-1)),
        ), dim=-1)

    @property
    def initial_state(self):
        if self.rb is None:
            raise RuntimeError('state must be initialized')
        return torch.cat((
            self.rb.origins.view(-1, self.rb.origins.size(-1)),
            self.initial_directions.view(-1, self.initial_directions.size(-1)),
        ), dim=-1)

    @initial_state.setter
    def initial_state(self, rb: RayBundle):
        self.rb = rb
        self.initial_directions = rb.directions

    def reset(self):
        # Reset the environment to an initial state
        self.rb.directions.copy_(self.initial_directions)
        return self.initial_state

    def eval_model(self, with_img=False):
        """Compute radiance at current state."""
        directions = self.rb.directions.clone() # camera coords
        new_directions = camera_to_world(
            self.rb.directions, self.rb.camera_to_world
        ) - self.rb.origins # world coords
        self.rb.directions = new_directions / new_directions.norm(dim=-1, keepdim=True)
        outputs = self.model(self.rb)
        self.rb.directions = directions # back to camera

        rgb = outputs['rgb'] # [0, 1]
        img = rgb.mean(dim=-1) \
            .reshape(-1, *self.rb.imshape) \
                .detach().cpu().numpy() if with_img else None

        if self.with_radiance:
            acc = outputs['accumulation'] # [0, 1]
            rad = rgb / acc.clip(min=1e-10) * self.rad_max # radiance
        else:
            rad = rgb * self.rad_max

        rad = rad.mean(dim=-1, keepdim=True) # grayscale
        rad = rad.clip(min=0, max=self.rad_max)
        return rad, img

    def step(self, action, with_img=False):
        if self.with_penalty:
            directions = self.rb.directions.clone()
        # take step by moving rays
        self.rb.directions.copy_(action)
        rad, img = self.eval_model(with_img)
        reward, penalty = self.reward_func(rad)
        above_thres = rad >= self.rad_thres
        done = above_thres.all().item()
        done_count = above_thres.sum().item()
        info = {
            'done': done_count / action.size(0),
            'img': img,
        }
        # freeze ray when penalty > 0
        if self.with_penalty:
            mask = (penalty > 0).repeat(
                *[1 for _ in range(penalty.ndim - 1)],
                directions.size(-1) // penalty.size(-1)
            )
            self.rb.directions[mask] = directions[mask]
            done = (penalty > 0).all().item()

        return self.state, reward, done, info

    def reward_func(self, rad):
        """reward is scaled radiance minus number of colliding rays"""
        penalty = torch.zeros(*rad.shape, device=rad.device)

        if self.with_penalty:
            dir_3d = self.directions.reshape(-1, self.imsize, self.directions.size(-1))
            rad_3d = rad.reshape(-1, self.imsize, rad.size(-1)) \
                .repeat(1, 1, self.imsize)

            dir_rep1 = dir_3d.unsqueeze(1).repeat(1, self.imsize, 1, 1)
            dir_rep2 = dir_3d.unsqueeze(2).repeat(1, 1, self.imsize, 1)
            dist = (dir_rep1 - dir_rep2).norm(dim=-1)
            # max number of rays per pixel is (max_resolution - 1)^2
            dist_min = self.px_dist / self.reward_max_resolution
            dist_thres = (self.px_dist * (self.rad_max - rad_3d) / self.rad_max).clip(min=dist_min)
            close_count = (
                (dist < dist_thres).float().sum(dim=-1, keepdim=True) - 1
            ).flatten(end_dim=1)

            mask = torch.logical_and(close_count > 0, rad >= self.rad_thres)
            penalty[mask] = close_count[mask]

        reward = ((rad - self.rad_thres) / self.rad_max - penalty) * self.reward_scale
        return reward, penalty
    
    def eval_actions(self, actions):
        """Compute radiance resulting from the given set of actions"""
        rb = deepcopy(self.rb)
        num_repeats = actions.size(0) // rb.size
        # bring RayBundle to desired shape
        self.rb.origins = self.rb.origins \
            .unsqueeze(1).repeat(1, num_repeats, 1).flatten(end_dim=1)
        self.rb.camera_indices = self.rb.camera_indices \
            .unsqueeze(1).repeat(1, num_repeats, 1).flatten(end_dim=1)
        self.rb.metadata['directions_norm'] = self.rb.metadata['directions_norm'] \
            .unsqueeze(1).repeat(1, num_repeats, 1).flatten(end_dim=1)
        self.rb.pixel_area = self.rb.pixel_area \
            .unsqueeze(1).repeat(1, num_repeats, 1).flatten(end_dim=1)
        self.rb._shape = actions.shape[:-1]
        self.rb.batch_size *= num_repeats
        self.rb.camera_to_world = self.rb.camera_to_world \
            .repeat_interleave(num_repeats, dim=0)
        self.rb.camera_intrinsics = self.rb.camera_intrinsics \
            .repeat_interleave(num_repeats, dim=0)

        self.rb.directions = actions
        rad, _  = self.eval_model()

        self.rb = rb
        return rad
