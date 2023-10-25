import torch
import matplotlib.pyplot as plt
from .geom import (
    camera_to_world,
    world_to_image,
)


def pre_render(model, dataloader, database, plot=False):
    for rb in dataloader:
        directions = rb.directions.clone()
        new_directions = camera_to_world(rb.directions, rb.camera_to_world) - rb.origins
        rb.directions = new_directions / new_directions.norm(dim=-1, keepdim=True)
        model_outputs = model(rb)
        rb.directions = directions
        rgb = model_outputs['rgb'].detach()
        acc = model_outputs['accumulation'].detach()

        for i in range(0, rb.imsize*rb.batch_size, rb.imsize):
            origin = tuple(rb.origins[i].cpu().tolist())
            rgb_ = rgb[i:i+rb.imsize]
            acc_ = acc[i:i+rb.imsize]
            assert rb.imsize == rgb_.shape[0]
            assert rb.imsize == acc_.shape[0]
            database[origin] = {
                'rgb': rgb_,
                'accumulation': acc_,
                'imshape': rb.imshape,
                'imsize': rb.imsize,
            }
            if plot:
                plt.figure()
                img = rgb_.reshape(*rb.imshape, rgb.size(-1)).cpu().numpy()
                plt.imshow(img)


class CustomNeRF:
    def __init__(self, db):
        self.db = db

    def __call__(self, rb):
        out = dict()
        rgb_ = torch.tensor([], device=rb.device)
        step = (rb.origins == rb.origins[0]).all(dim=-1).sum(dim=-1).item()
        # TODO: accumulation
        for i in range(0, rb.size, step):
            # retrieve image from db
            origin = tuple(rb.origins[i].cpu().tolist())
            img = self.db[origin]
            # select portion of image according to rb shape
            rgb = img['rgb'].reshape(*img['imshape'], -1)
            rgb = rgb[:rb.imshape[0], :rb.imshape[1]]
            # find intersection of ray directions with image plane
            points = rb.origins[i:i+step] + rb.directions[i:i+step]
            img_coords = world_to_image(
                points,
                rb.camera_to_world[i//step:i//step + 1],
                rb.camera_intrinsics[i//step:i//step + 1],
            ).reshape(step, -1)
            # compute rgb values by bilinear interpolation
            upper_left = img_coords.round().int() - 1 # upper-left of 4 nearest neighbors
            local = img_coords - (upper_left + 0.5) # local coordinates
            k = upper_left[..., 0]
            l = upper_left[..., 1]
            u = local[..., 0].unsqueeze(-1)
            v = local[..., 1].unsqueeze(-1)
            top_left = rgb[k % rb.imshape[0], l % rb.imshape[1]] # RGB values
            bottom_left = rgb[(k+1) % rb.imshape[0], l % rb.imshape[1]]
            top_right = rgb[k % rb.imshape[0], (l+1) % rb.imshape[1]]
            bottom_right = rgb[(k+1) % rb.imshape[0], (l+1) % rb.imshape[1]]
            _rgb_ = (
                (1-v)*((1-u)*top_left + u*bottom_left) + v*((1-u)*top_right + u*bottom_right)
            ).reshape(step, -1)
            assert not _rgb_.isnan().any()
            # concatenate results
            rgb_ = torch.cat((rgb_, _rgb_))
        out['rgb'] = rgb_
        return out
