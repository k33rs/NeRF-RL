import torch
from typing import Callable
from .geom import world_to_camera


class CustomDL:
    def __init__(
            self,
            next_call: Callable,
            cameras=None,
            step_size=1,
            device='cpu',
    ):
        self.next_call = next_call
        self.cameras = cameras
        self.step = 0
        self.step_size = step_size
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        if self.step >= len(self.cameras):
            self.reset()
            raise StopIteration
        batch = self[self.step]
        self.step += self.step_size
        return batch
    
    def __getitem__(self, step):
        return self.next_call(self.cameras, range(step, step+self.step_size), self.device)
    
    def __len__(self):
        return len(self.cameras)
    
    def reset(self, step=0):
        self.step = step


def next_batch(cameras, indices, device='cpu', debug=False):
    cameras = cameras[torch.tensor(indices)]
    rb = cameras.generate_rays(torch.tensor([[i] for i in range(len(indices))]))
    # flatten the RayBundle
    imshape = rb.shape[:2]
    batch_size = rb.shape[-1]
    rb = rb.reshape((-1, batch_size))
    rb.origins = rb.origins.transpose(0, 1)
    rb.directions = rb.directions.transpose(0, 1)
    rb.camera_indices = rb.camera_indices.transpose(0, 1)
    rb.metadata['directions_norm'] = rb.metadata['directions_norm'].transpose(0, 1)
    rb.pixel_area = rb.pixel_area.transpose(0, 1)
    rb._shape = rb.directions.shape[:-1]
    rb = rb.flatten().to(device)
    rb.imshape = imshape
    rb.imsize = torch.tensor(imshape).prod().item()
    rb.batch_size = batch_size
    rb.device = rb.origins.device
    # keep camera parameters for mapping world to image coordinates
    rb.camera_to_world = cameras.camera_to_worlds.to(device)
    rb.camera_intrinsics = cameras.get_intrinsics_matrices().to(device)
    # convert directions to camera coordinates
    rb.directions = world_to_camera(rb.origins + rb.directions, rb.camera_to_world)
    rb.directions /= rb.directions.norm(dim=-1, keepdim=True)
    # angle between ray and bottom-right neighboring ray
    rb.clip_angle = torch.acos(
        torch.tensor(
            [(rb.directions[0] @ rb.directions[imshape[1] + 1]).clip(-1, 1)],
            device=device,
        )
    )
    rb.clip_angle_small = rb.clip_angle / 8
    # distance between rays passing through pixel centers (for reward function)
    rb.px_dist = (rb.directions[0] - rb.directions[1]).norm(dim=-1)

    if debug:
        size_bytes = rb.origins.element_size() * rb.origins.nelement() \
            + rb.directions.element_size() * rb.directions.nelement() \
            + rb.camera_indices.element_size() * rb.camera_indices.nelement() \
            + rb.metadata['directions_norm'].element_size() * rb.metadata['directions_norm'].nelement() \
            + rb.pixel_area.element_size() * rb.pixel_area.nelement() \
            + rb.fov_axis.element_size() * rb.fov_axis.nelement() \
            + rb.fov.element_size() * rb.fov.nelement()
        print('camera indices:', indices, 'size (bytes):', size_bytes)

    return rb
