import numpy as np
import torch
from typing import Literal


def random_dir(
        dir: torch.Tensor,
        angle_rad: torch.Tensor,
        angle_sample: Literal['uniform', 'normal'] = 'uniform',
) -> torch.Tensor:
    """
    Random unit direction around the given direction within the given angle limit.
    """
    # rotation axis (uniform sampling on the unit sphere)
    axis = 2 * torch.rand(*dir.shape, device=dir.device) - 1 # [-1, 1]
    axis = axis / axis.norm(dim=-1, keepdim=True)
    dir = dir / dir.norm(dim=-1, keepdim=True)
    # rotation angle
    num_rays = dir.size(0)
    if angle_sample == 'uniform':
        theta = torch.rand(num_rays, device=dir.device) * angle_rad
    elif angle_sample == 'normal':
        theta = torch.randn(num_rays, device=dir.device).abs() * angle_rad
    # rotation matrix
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]

    xx = (1-cos_theta)*x*x + cos_theta
    xy = (1-cos_theta)*x*y - sin_theta*z
    xz = (1-cos_theta)*x*z + sin_theta*y
    yx = (1-cos_theta)*x*y + sin_theta*z
    yy = (1-cos_theta)*y*y + cos_theta
    yz = (1-cos_theta)*y*z - sin_theta*x
    zx = (1-cos_theta)*x*z - sin_theta*y
    zy = (1-cos_theta)*y*z + sin_theta*x
    zz = (1-cos_theta)*z*z + cos_theta

    R = torch.stack([
        torch.stack([xx, xy, xz], dim=1),
        torch.stack([yx, yy, yz], dim=1),
        torch.stack([zx, zy, zz], dim=1)
    ], dim=1)
    new_dir = (R @ dir.unsqueeze(-1)).squeeze(-1)
    # assert (torch.acos((dir*new_dir).sum(dim=-1).clip(-1, 1)) <= angle_rad).all()
    return new_dir


def clip_to_angle(
        dir: torch.Tensor,
        axis: torch.Tensor,
        clip_angle_rad: torch.Tensor,
        ignore: torch.Tensor = None,
) -> torch.Tensor:
    """
    Clip directions within a given angle w.r.t. axis
    """
    dir = dir / dir.norm(dim=-1, keepdim=True)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    # angles between the vectors and the axis before rotation
    dot = (axis * dir).sum(dim=-1)
    angle_rad = torch.acos(dot.clip(-1, 1))
    # rotation axis perpendicular to the axis and vectors
    rot_axis = torch.cross(axis, dir)
    rot_axis = rot_axis / rot_axis.norm(dim=-1, keepdim=True)
    u = rot_axis.unsqueeze(-1)
    uuT = u @ u.transpose(-2, -1) # outer product
    # rotate vectors that need it
    cos_theta = torch.cos(clip_angle_rad).to(dir.device)
    sin_theta = torch.sin(clip_angle_rad).to(dir.device)
    num_vec = torch.tensor(axis.shape[:-1]).prod().item()
    I = torch.stack([torch.eye(3)] * num_vec) \
        .view(*axis.shape[:-1], 3, 3).to(dir.device)
    # Rodriguez rotation formula
    R = cos_theta * I + (1 - cos_theta) * uuT \
        + sin_theta * cross_prod_matrix(rot_axis)
    # combine rotated and unrotated vectors
    rotated = (R @ axis.unsqueeze(-1)).squeeze(-1)
    mask = (angle_rad > clip_angle_rad).unsqueeze(-1)
    if ignore is not None:
        mask = torch.logical_and(mask, ~ignore)
    clip = dir * ~mask + rotated * mask
    # normalize (happens only if direction is antipodal w.r.t. hemi_axis)
    norm, one = clip.norm(dim=-1), torch.ones(1, device=dir.device)
    if not torch.isclose(norm, one).all():
        clip = clip / clip.norm(dim=-1, keepdim=True)
    return clip


def cross_prod_matrix(v: torch.Tensor) -> torch.Tensor:
    """
    Construct a cross-product matrix for vectors v.
    """
    zeros = torch.zeros_like(v[..., 0])
    return torch.stack([
        zeros, -v[..., 2], v[..., 1],
        v[..., 2], zeros, -v[..., 0],
        -v[..., 1], v[..., 0], zeros
    ], dim=-1).view(*v.shape[:-1], 3, 3)


def clip_to_image(
        dir: torch.Tensor,
        default: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        imshape: torch.Size,
) -> torch.Tensor:
    """
    Clip camera directions to image boundaries.
    """
    img_coords = camera_to_image(dir, camera_intrinsics)
    to_clip = torch.logical_or(
        torch.logical_or(img_coords[..., 0] < 0, img_coords[..., 0] >= imshape[0]),
        torch.logical_or(img_coords[..., 1] < 0, img_coords[..., 1] >= imshape[1]),
    )

    if not to_clip.any().item():
        return dir

    x = img_coords[to_clip][..., 0].clip(0, imshape[0] - 1).int()
    y = img_coords[to_clip][..., 1].clip(0, imshape[1] - 1).int()

    imsize = torch.tensor(imshape).prod()
    img_idx = torch.arange(dir.size(-2), device=dir.device) // imsize
    if img_idx.ndim < to_clip.ndim:
        img_idx = img_idx.unsqueeze(0).repeat(to_clip.size(0), 1)
    img_offset = img_idx[to_clip] * imsize

    with torch.no_grad():
        dir[to_clip] = default[img_offset + x * imshape[1] + y]
    return dir


def camera_to_world(
        camera: torch.Tensor,
        camera2world: torch.Tensor,
) -> torch.Tensor:
    """
    Convert from camera to world coordinates
    """
    camera_size = torch.tensor(camera.shape[:-1]).prod().item()
    pointsxcam = np.max([1, camera_size // camera2world.size(0)])

    last_row = torch.tensor([[[0, 0, 0, 1]]], device=camera2world.device)
    c2w_homo = torch.cat((
        camera2world,
        last_row.repeat(camera2world.size(0), 1, 1),
    ), dim=1)
    c2w_rep = c2w_homo.repeat_interleave(pointsxcam, dim=0) \
        .reshape(*camera.shape[:-1], *c2w_homo.shape[-2:])
    camera_homo = torch.cat((
        camera,
        torch.ones(*camera.shape[:-1], 1, device=camera.device)
    ), dim=-1)

    world = (
        c2w_rep @ camera_homo.unsqueeze(-1)
    ).squeeze(-1)
    # preserve gradient computation (no in-place operations)
    world = world.clone() / world[..., 3].unsqueeze(-1)
    return world[..., :3]


def world_to_camera(
        world: torch.Tensor,
        camera2world: torch.Tensor,
) -> torch.Tensor:
    """
    Convert from world to camera coordinates.
    """
    world_size = torch.tensor(world.shape[:-1]).prod().item()
    pointsxcam = np.max([1, world_size // camera2world.size(0)])

    last_row = torch.tensor([[[0, 0, 0, 1]]], device=camera2world.device)
    extrinsics = torch.linalg.inv(
        torch.cat((
            camera2world,
            last_row.repeat(camera2world.size(0), 1, 1),
        ), dim=1)
    )
    extrinsics_rep = extrinsics.repeat_interleave(pointsxcam, dim=0) \
        .reshape(*world.shape[:-1], *extrinsics.shape[-2:])
    world_homo = torch.cat((
        world,
        torch.ones(*world.shape[:-1], 1, device=world.device)
    ), dim=-1)

    camera = (
        extrinsics_rep @ world_homo.unsqueeze(-1)
    ).squeeze(-1)
    # preserve gradient computation (no in-place operations)
    camera = camera.clone() / camera[..., 3].unsqueeze(-1)
    return camera[..., :3]


def camera_to_image(
        camera: torch.Tensor,
        intrinsics: torch.Tensor,
) -> torch.Tensor:
    """
    Convert from camera to image coordinates
    """
    camera_size = torch.tensor(camera.shape[:-1]).prod().item()
    pointsxcam = np.max([1, camera_size // intrinsics.size(0)])

    intrinsics_rep = intrinsics.repeat_interleave(pointsxcam, dim=0) \
        .reshape(*camera.shape[:-1], *intrinsics.shape[-2:])
    rotate = torch.tensor(
        [[[0, 1, 0], [-1, 0, 0], [0, 0, 1]]],
        dtype=torch.float32,
        device=camera.device
    ) # cw rotation by pi
    rotate_rep = rotate.repeat(camera_size, 1, 1) \
        .reshape(*camera.shape[:-1], *rotate.shape[-2:])
    img = (
        intrinsics_rep @ rotate_rep @ camera.unsqueeze(-1)
    ).squeeze(-1)
    # preserve gradient computation (no in-place operations)
    img = img.clone() / img[..., 2].unsqueeze(-1)
    return img[..., :2]


def world_to_image(
        world: torch.Tensor,
        camera2world: torch.Tensor,
        intrinsics: torch.Tensor,
) -> torch.Tensor:
    """
    Convert from world to image coordinates
    """
    camera = world_to_camera(world, camera2world)
    img = camera_to_image(camera, intrinsics)
    return img
