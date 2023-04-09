"""dataloader.py."""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils import generate_rays


class NeRFDataset(Dataset):
    """Custom dataset class for loading data, generating rays and sampling points."""

    def __init__(self, dataset_path, image_size, camera_matrix, num_sample_points):
        """
        Initialize the dataset.

        :param dataset_path: str, path to the dataset
        :param image_size: tuple, (height, width) of the input images
        :param camera_matrix: numpy array, 3x4 camera matrix
        :param num_sample_points: int, number of sample points along each ray
        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.camera_matrix = camera_matrix
        self.num_sample_points = num_sample_points

        self.image_paths = sorted(
            [
                os.path.join(self.dataset_path, img)
                for img in os.listdir(self.dataset_path)
                if img.endswith(".png") or img.endswith(".jpg")
            ]
        )

    def __len__(self):
        """
        Return the number of images in the dataset.

        :return: int, number of images
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load an image, generate rays, and sample points along the rays.

        :param idx: int, index of the image
        :return: dict, contains the image tensor and the sample points tensor
        """
        img = Image.open(self.image_paths[idx])
        img = img.resize(self.image_size)
        img = np.array(img, dtype=np.float32) / 255.0

        H, W, _ = img.shape  # noqa
        ray_origins, ray_directions = generate_rays(self.camera_matrix, H, W)

        t_vals = torch.linspace(0, 1, steps=self.num_sample_points)
        t_vals = t_vals.view(1, 1, 1, -1).expand(H, W, 3, -1)

        ray_origins = ray_origins.unsqueeze(-1).expand_as(t_vals)
        ray_directions = ray_directions.unsqueeze(-1).expand_as(t_vals)

        sample_points = ray_origins + t_vals * ray_directions
        sample_points = sample_points.view(-1, 3)

        return {
            "image": torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32),
            "sample_points": sample_points,
        }


def create_nerf_dataloader(
    dataset_path, image_size, camera_matrix, num_sample_points, batch_size
):
    """
    Create a DataLoader for the NeRF dataset.

    :param dataset_path: str, path to the dataset
    :param image_size: tuple, (height, width) of the input images
    :param camera_matrix: numpy array, 3x4 camera matrix
    :param num_sample_points: int, number of sample points along each ray
    :param batch_size: int, batch size for the DataLoader
    :return: DataLoader, the NeRF dataset DataLoader
    """
    dataset = NeRFDataset(dataset_path, image_size, camera_matrix, num_sample_points)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
