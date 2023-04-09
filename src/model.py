"""model.py."""
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa


class PositionalEncoding(nn.Module):
    """Custom class for the positional encodings."""

    def __init__(self, L):  # noqa
        """
        Construct the PositionalEncoding class.

        Args:
        - L (int): The number of encoding functions to use.
        """
        super(PositionalEncoding, self).__init__()
        self.L = L  # noqa

    def forward(self, x):
        """Forward pass of the PositionalEncoding module.

        Args:
        - x (torch.Tensor): The input tensor to be encoded.

        Returns:
        - out (torch.Tensor): The encoded tensor.
        """
        out = [x]
        for i in range(self.L):
            for fn in (torch.sin, torch.cos):
                out.append(fn(2.0**i * x))
        return torch.cat(out, dim=-1)


class NeRFModel(nn.Module):
    """Custom NeRFModel class."""

    def __init__(self, num_layers=8, num_features=256, L_pos=6, L_dir=4):  # noqa
        """
        Construct the NeRFModel.

        Args:
        - num_layers (int): The number of fully connected layers in the model.
        - num_features (int): The number of features in each layer.
        - L_pos (int): Number of encoding functions to use for the positional encoding.
        - L_dir (int): Number of encoding functions to use for the directional encoding.
        """
        super(NeRFModel, self).__init__()

        self.positional_encoding = PositionalEncoding(L_pos)
        self.directional_encoding = PositionalEncoding(L_dir)

        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            in_features = num_features * (2 if i == 0 else 1)
            out_features = num_features * 2 if i < num_layers - 2 else 4
            self.layers.append(nn.Linear(in_features, out_features))

        self.alpha_layer = nn.Linear(num_features, 1)
        self.rgb_layer = nn.Linear(num_features, 3)

    def forward(self, x, view_direction):
        """Forward pass of the NeRFModel module.

        Args:
        - x (torch.Tensor): The input tensor of sample points.
        - view_direction (torch.Tensor): The view direction for each sample point.

        Returns:
        - alpha (torch.Tensor): The predicted density of each sample point.
        - rgb (torch.Tensor): The predicted color of each sample point.
        """
        x = self.positional_encoding(x)
        view_direction = self.directional_encoding(view_direction)

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
            if i == len(self.layers) // 2 - 1:
                x = torch.cat([x, view_direction], dim=-1)

        alpha = F.relu(self.alpha_layer(x))
        rgb = torch.sigmoid(self.rgb_layer(x))

        return alpha, rgb
