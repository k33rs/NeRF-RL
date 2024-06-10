import gym
import numpy as np
import torch
from gym import spaces
from copy import deepcopy
from typing import TypeVar, Callable
from .geom import camera_to_world


RayBundle = TypeVar('RayBundle')


class NerfEnv(gym.Env):
    def __init__(
            self,
            model,
            reward_func: Callable,
            reward_thres: float=1.,
            reward_max: float=1.,
            reward_scale: float=1.,
            with_radiance=False
    ):
        super(NerfEnv, self).__init__()
        # Initialize environment
        self.rb = None # RayBundle
        self.initial_directions = None
        self.reward_func = reward_func
        self.reward_thres = reward_thres
        self.reward_max = reward_max
        self.reward_scale = reward_scale
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
        gs = rgb.mean(dim=-1)
        img = gs.reshape(rgb.shape[0] // self.rb.imsize, *self.rb.imshape) \
            .cpu().numpy() if with_img else None

        if self.with_radiance:
            acc = outputs['accumulation'] # [0, 1]
            rad = rgb / acc.clip(min=1e-10) * self.reward_max # radiance
        else:
            rad = rgb * self.reward_max
        rad = rad.mean(dim=-1).unsqueeze(-1) \
            .clip(min=0, max=self.reward_max) # grayscale

        return rad, img
    
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

    def step(self, action, with_img=False):
        # Take a step in the environment based on the given action
        self.rb.directions.copy_(action)
        rad, img = self.eval_model(with_img)
        # next_state, reward, done
        reward = self.reward_func(rad, self.rb.directions, self.rb.imsize) \
            * self.reward_scale
        above_thres = rad >= self.reward_thres
        done = above_thres.all().item()
        done_count = above_thres.sum().item()
        info = {
            'done_count': done_count,
            'done_ratio': done_count / action.size(0),
            'img': img,
        }

        return self.state, reward, done, info
