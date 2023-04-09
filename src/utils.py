import torch

def generate_rays(camera_matrix, H, W):
    # Create a meshgrid of pixel coordinates
    x = torch.linspace(0, W - 1, W, dtype=torch.float32)
    y = torch.linspace(0, H - 1, H, dtype=torch.float32)
    xv, yv = torch.meshgrid(x, y)
    pixel_coords = torch.stack([xv, yv], dim=-1)

    # Compute ray directions in the camera coordinate system
    inv_camera_matrix = torch.inverse(camera_matrix)
    homogeneous_pixel_coords = torch.cat([pixel_coords, torch.ones_like(pixel_coords[..., :1])], dim=-1)
    ray_directions = torch.matmul(homogeneous_pixel_coords, inv_camera_matrix.t())[:, :, :3]

    # Normalize ray directions
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

    # Compute ray origins in the camera coordinate system
    ray_origins = torch.zeros_like(ray_directions)

    return ray_origins, ray_directions

def raymarch(sample_points, view_directions, model):
    alpha, rgb = model(sample_points, view_directions)

    # Compute accumulated transmittance
    accumulated_transmittance = torch.exp(-torch.cumsum(alpha, dim=1))

    # Compute the color at each sampled point
    colors = (1 - torch.exp(-alpha)).unsqueeze(-1) * rgb

    # Accumulate colors along the ray
    accumulated_colors = torch.cumsum(colors * accumulated_transmittance.unsqueeze(-1), dim=1)

    return accumulated_colors

def compute_loss(accumulated_colors, img, config):
    H, W, _ = img.shape

    # Compute the estimated image
    estimated_img = torch.sum(accumulated_colors, dim=1)
    estimated_img = estimated_img.view(H, W, 3)

    # Compute the loss between the estimated image and the ground truth image
    loss = torch.mean((estimated_img - img) ** 2)

    return loss
