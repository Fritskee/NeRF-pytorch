"""inference.py."""
import torch
from PIL import Image

from model import NeRFModel
from utils import generate_rays, raymarch


def load_trained_model(model_checkpoint_path):
    """
    Load a trained NeRF model from a checkpoint.

    :param model_checkpoint_path: str, path to the model checkpoint file
    :return: NeRFModel, trained NeRF model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeRFModel().to(device)
    model.load_state_dict(torch.load(model_checkpoint_path))
    model.eval()
    return model


def render_image(model, camera_matrix, image_size, num_sample_points):
    """
    Render an image using a trained NeRF model.

    :param model: NeRFModel, trained NeRF model
    :param camera_matrix: numpy array, 3x4 camera matrix
    :param image_size: tuple, (height, width) of the output image
    :param num_sample_points: int, number of sample points along each ray
    :return: torch.Tensor, the rendered image
    """
    H, W = image_size  # noqa
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate rays
    ray_origins, ray_directions = generate_rays(camera_matrix, H, W)
    ray_origins = ray_origins.to(device)
    ray_directions = ray_directions.to(device)

    # Sample points along the rays
    t_vals = torch.linspace(0, 1, steps=num_sample_points)
    t_vals = t_vals.view(1, 1, 1, -1).expand(H, W, 3, -1)

    ray_origins = ray_origins.unsqueeze(-1).expand_as(t_vals)
    ray_directions = ray_directions.unsqueeze(-1).expand_as(t_vals)

    sample_points = ray_origins + t_vals * ray_directions
    sample_points = sample_points.view(-1, 3)

    # Generate view directions
    view_directions = torch.randn_like(
        sample_points
    )  # Use random view directions for this example

    # Perform ray marching
    accumulated_colors = raymarch(sample_points, view_directions, model)
    accumulated_colors = accumulated_colors.view(H, W, -1, 3)

    # Assemble the output image
    output_img = torch.sum(accumulated_colors, dim=2)

    return output_img


def inference(config):
    """
    Perform inference using a trained NeRF model and save the rendered image.

    :param config: Config, configuration object containing model and image settings
    """
    # Load trained NeRF model
    model = load_trained_model(config.model_checkpoint_path)

    # Render image using the trained model
    output_img = render_image(
        model,
        config.camera_matrix,
        (config.image_height, config.image_width),
        config.num_sample_points,
    )

    # Save the output image
    output_img = (output_img.detach().cpu().numpy() * 255).astype("uint8")
    Image.fromarray(output_img).save(config.output_image_path)
