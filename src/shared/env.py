import gym
import numpy as np
import torch
from gym import spaces
from copy import deepcopy
from typing import TypeVar
from .geom import camera_to_world, camera_to_image


RayBundle = TypeVar('RayBundle')


class NerfEnv(gym.Env):
    def __init__(
            self,
            model,
            rad_thres: float=1.,
            rad_max: float=1.,
            reward_scale: float=1.,
            reward_thres: float=1.,
            reward_max_resolution: int=1,
            is_training=False,
            with_radiance=False
    ):
        super(NerfEnv, self).__init__()
        # Initialize environment
        self.rb = None # RayBundle
        self.initial_directions = None
        self.rad_thres = rad_thres
        self.rad_max = rad_max
        self.reward_scale = reward_scale
        self.reward_thres = reward_thres
        self.reward_max_resolution = reward_max_resolution
        self.is_training = is_training
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
        self._img = None # managed by runner
        self.ray_counts = None

    def init_img(self):
        if self.is_training:
            return

        self._img = torch.zeros(
            (self.rb.batch_size, *self.rb.imshape),
            device=self.rb.device
        )
        self.ray_counts = torch.zeros(
            (self.rb.batch_size, *self.rb.imshape),
            device=self.rb.device,
            dtype=torch.int
        )

    @property
    def img(self):
        return self._img.cpu().numpy()

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
        self.init_img()
        return self.initial_state

    def eval_model(self):
        """Compute radiance at current state."""
        directions = self.rb.directions.clone()
        self.rb.directions = camera_to_world(
            self.rb.directions, self.rb.camera_to_world
        ) - self.rb.origins
        self.rb.directions /= self.rb.directions.norm(dim=-1, keepdim=True)
        outputs = self.model(self.rb)
        self.rb.directions = directions

        rgb = outputs['rgb'].detach() # [0, 1]
        img = rgb.mean(dim=-1).view(self.rb.batch_size, *self.rb.imshape, -1)
        self.update_img(img)
        img = img.squeeze(-1).cpu().numpy()

        if self.with_radiance:
            acc = outputs['accumulation'] # [0, 1]
            rad = rgb / acc.clip(min=1e-10) * self.rad_max # radiance
        else:
            rad = rgb * self.rad_max

        rad = rad.mean(dim=-1, keepdim=True) # grayscale
        rad = rad.clip(min=0, max=self.rad_max)
        return rad, img

    def step(self, action):
        if self.with_penalty:
            directions = self.rb.directions.clone()
        # take step by moving rays
        self.rb.directions.copy_(action)
        rad, img = self.eval_model()
        reward, penalty = self.reward_func(rad)
        above_thres = rad >= self.rad_thres
        done = above_thres.all().item()
        done_count = above_thres.sum().item()
        info = {
            'done': done_count / action.size(0),
            'img': img if self.is_training else self.img,
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
            dir_3d = self.rb.directions.view(-1, self.rb.imsize, self.rb.directions.size(-1))
            rad_3d = rad.view(-1, self.rb.imsize, rad.size(-1)) \
                .repeat(1, 1, self.rb.imsize)

            dir_rep1 = dir_3d.unsqueeze(1).repeat(1, self.rb.imsize, 1, 1)
            dir_rep2 = dir_3d.unsqueeze(2).repeat(1, 1, self.rb.imsize, 1)
            dist = (dir_rep1 - dir_rep2).norm(dim=-1)
            # max number of rays per pixel is max_resolution^2
            dist_min = self.rb.px_dist / (self.reward_max_resolution + 1)
            dist_thres = (self.rb.px_dist * (self.rad_max - rad_3d) / self.rad_max).clip(min=dist_min)
            close_count = (
                (dist < dist_thres).float().sum(dim=-1, keepdim=True) - 1
            ).flatten(end_dim=1)

            mask = torch.logical_and(close_count > 0, rad >= self.rad_thres)
            penalty[mask] = close_count[mask]

        reward = ((rad - self.reward_thres) / self.rad_max - penalty) * self.reward_scale
        return reward, penalty
    
    def update_img(self, img):
        # accumulate results into the rightful pixels
        # for i, batch in enumerate(img_coords):
        #     for row in batch:
        #         for x, y in row:
        #             n = self.ray_counts[i, x, y]
        #             self._img[i, x, y] = (self._img[i, x, y] * n + raw_img[i, x, y]) / (n + 1)
        #             self.ray_counts[i, x, y] = n + 1
        if self.is_training:
            return

        img_coords = camera_to_image(
            self.rb.directions, self.rb.camera_intrinsics
        ).floor().int().view(self.rb.batch_size, *self.rb.imshape, -1)

        img_idx = torch.arange(img_coords.size(0), device=img_coords.device) \
            .view(-1, 1, 1).expand_as(img_coords[..., 0]).flatten()
        row_idx = img_coords[..., 0].flatten()
        col_idx = img_coords[..., 1].flatten()

        flat_idx = img_idx * self.rb.imsize + row_idx * self.rb.imshape[1] + col_idx

        counts_up = self.ray_counts.flatten()

        img_up = self._img.flatten() * counts_up
        img_up.scatter_add_(dim=0, index=flat_idx, src=img.flatten())

        counts_up.scatter_add_(dim=0, index=flat_idx, src=torch.ones_like(counts_up))
        img_up /= counts_up

        self._img = img_up.view_as(self._img)
        self.ray_counts = counts_up.view_as(self.ray_counts)
    
    def get_density(self, desired=False):
        res = (
            self._img / self._img.sum(dim=(-2, -1), keepdim=True)
        ) if desired else (
            self.ray_counts / self.ray_counts.sum(dim=(-2, -1), keepdim=True)
        )
        return res.cpu().numpy()
    
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
