#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import math
import torch
from diff_gaussian_rasterization_32d import GaussianRasterizationSettings, GaussianRasterizer

NUM_CHANNELS = 32

def render_gaussian(gs_params, cam_matrix, cam_params=None, sh_degree=0, bg_color=None, projection_mode='perspective'):
    # Build params
    batch_size = cam_matrix.shape[0]
    focal_x, focal_y, cam_size = cam_params['focal_x'], cam_params['focal_y'], cam_params['size']
    points, colors, opacities, scales, rotations = \
        gs_params['xyz'], gs_params['colors'], gs_params['opacities'], gs_params['scales'], gs_params['rotations']
    view_mat, proj_mat, cam_pos = build_camera_matrices(cam_matrix, focal_x, focal_y, projection_mode)
    bg_color = cam_matrix.new_zeros(batch_size, NUM_CHANNELS, dtype=torch.float32) if bg_color is None else bg_color
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    means2D = torch.zeros_like(points, dtype=points.dtype, requires_grad=True, device="cuda") + 0
    try:
        means2D.retain_grad()
    except:
        pass
    # Run rendering
    all_rendered, all_radii = [], []
    for bid in range(batch_size):
        raster_settings = GaussianRasterizationSettings(
            sh_degree=sh_degree, bg=bg_color, 
            image_height=cam_size[0], image_width=cam_size[1],
            tanfovx=1.0 / focal_x, tanfovy=1.0 / focal_y,
            viewmatrix=view_mat[bid], projmatrix=proj_mat[bid], campos=cam_pos[bid],
            scale_modifier=1.0, prefiltered=False, debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rendered, radii = rasterizer(
            means3D=points[bid], means2D=means2D[bid], 
            shs=None, colors_precomp=colors[bid], 
            opacities=opacities[bid], scales=scales[bid], 
            rotations=rotations[bid], cov3D_precomp=None
        )
        all_rendered.append(rendered)
        all_radii.append(radii)
    all_rendered = torch.stack(all_rendered, dim=0)
    all_radii = torch.stack(all_radii, dim=0)
    return {
        "images": all_rendered, "radii": all_radii, "viewspace_points": means2D,
    }


def build_camera_matrices(cam_matrix, focal_x, focal_y, projection_mode='perspective'):
    def get_projection_matrix(fov_x, fov_y, z_near=0.01, z_far=100, device='cpu'):
        K = torch.zeros(4, 4, device=device)
        z_sign = 1.0
        K[0, 0] = 1.0 / math.tan((fov_x / 2))
        K[1, 1] = 1.0 / math.tan((fov_y / 2))
        K[3, 2] = z_sign
        K[2, 2] = z_sign * z_far / (z_far - z_near)
        K[2, 3] = -(z_far * z_near) / (z_far - z_near)
        return K

    def get_orthographic_projection_matrix(scale_x, scale_y, z_near=0.01, z_far=100, device='cpu'):
        K = torch.zeros(4, 4, device=device)
        K[0, 0] = scale_x
        K[1, 1] = scale_y
        K[2, 2] = -2 / (z_far - z_near)
        K[2, 3] = -(z_far + z_near) / (z_far - z_near)
        K[3, 3] = 1.0
        return K

    def get_world_to_view_matrix(transforms):
        assert transforms.shape[-2:] == (3, 4)
        viewmatrix = transforms.new_zeros(transforms.shape[0], 4, 4)
        for i in range(4):
            viewmatrix[:, i, i] = 1.0
        viewmatrix[:, :3, :3] = transforms[:, :3, :3]
        viewmatrix[:, 3, :3] = transforms[:, :3, 3]
        viewmatrix[:, :, :2] *= -1.0
        return viewmatrix

    def get_full_projection_matrix(viewmatrix, fov_x, fov_y, projection_mode, focal_x, focal_y):
        if projection_mode == 'orthographic':
            # In orthographic mode, focal length acts as a scaling factor.
            # A higher focal length means a smaller object (zoomed out).
            proj_matrix = get_orthographic_projection_matrix(focal_x / 10.0, focal_y / 10.0, device=viewmatrix.device)
        else: # perspective
            proj_matrix = get_projection_matrix(fov_x, fov_y, device=viewmatrix.device)
        
        full_proj_matrix = viewmatrix @ proj_matrix.transpose(0, 1)
        return full_proj_matrix

    fov_x = 2 * math.atan(1.0 / focal_x)
    fov_y = 2 * math.atan(1.0 / focal_y)
    view_matrix = get_world_to_view_matrix(cam_matrix)
    full_proj_matrix = get_full_projection_matrix(view_matrix, fov_x, fov_y, projection_mode, focal_x, focal_y)
    cam_pos = cam_matrix[:, :3, 3]
    return view_matrix, full_proj_matrix, cam_pos


def create_background_gaussians(background_image, blur_level=0.5, z_depth=10.0, cam_params=None, device='cpu'):
    """
    Converts a 2D image into a 3D Gaussian cloud representation for background rendering.

    Args:
        background_image (torch.Tensor): The background image tensor (C, H, W).
        blur_level (float): Controls the amount of blur by scaling the Gaussians.
        z_depth (float): The distance of the background plane from the camera.
        cam_params (dict): Camera parameters including focal length.
        device (str): The device to create tensors on.

    Returns:
        dict: A dictionary containing the parameters for the background Gaussian cloud.
    """
    import torchvision.transforms.functional as F

    # 1. Determine the size of the 3D plane to match the camera's field of view
    focal_x = cam_params.get('focal_x', 12.0)
    focal_y = cam_params.get('focal_y', 12.0)
    
    # The width of the plane at distance z_depth is 2 * z_depth / focal_length
    plane_height = (2 * z_depth) / focal_y
    plane_width = (2 * z_depth) / focal_x

    # 2. Downsample the image to create a manageable number of Gaussians
    num_gaussians_h = 128  # Create a 128x128 grid of Gaussians
    num_gaussians_w = 128
    
    # Resize image and get color for each Gaussian
    bg_resized = F.resize(background_image, (num_gaussians_h, num_gaussians_w), antialias=True)
    colors = bg_resized.permute(1, 2, 0).reshape(-1, 3)  # (N, 3)
    # The GAGAvatar renderer uses a 32-channel feature space, so we pad the colors
    colors_padded = torch.zeros(colors.shape[0], 32, device=device)
    colors_padded[:, :3] = colors

    # 3. Create the grid of 3D points for the plane
    y = torch.linspace(plane_height / 2, -plane_height / 2, num_gaussians_h, device=device)
    x = torch.linspace(-plane_width / 2, plane_width / 2, num_gaussians_w, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # The z-coordinate is fixed
    grid_z = torch.full_like(grid_x, z_depth)
    
    xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3) # (N, 3)

    # 4. Define other Gaussian properties
    num_points = xyz.shape[0]
    
    # Opacity should be fully opaque
    opacities = torch.ones(num_points, 1, device=device)
    
    # Scale controls the blur. Base scale ensures Gaussians touch to form a solid plane.
    # Blur level increases the scale to create an out-of-focus effect.
    base_scale_x = (plane_width / num_gaussians_w) / 2
    base_scale_y = (plane_height / num_gaussians_h) / 2
    blur_multiplier = 1.0 + blur_level * 5.0 # Heuristic multiplier for blur
    scales = torch.zeros(num_points, 3, device=device)
    scales[:, 0] = base_scale_x * blur_multiplier
    scales[:, 1] = base_scale_y * blur_multiplier
    scales[:, 2] = 0.01 # Depth scale is small

    # Rotation is identity (no rotation)
    rotations = torch.zeros(num_points, 4, device=device)
    rotations[:, 0] = 1.0

    return {
        'xyz': xyz,
        'colors': colors_padded,
        'opacities': opacities,
        'scales': scales,
        'rotations': rotations,
    }