import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, L):
        super(PositionalEncoding, self).__init__()
        self.L = L

    def forward(self, x):
        out = [x]
        for i in range(self.L):
            for fn in (torch.sin, torch.cos):
                out.append(fn(2.0 ** i * x))
        return torch.cat(out, dim=-1)

class NeRFModel(nn.Module):
    def __init__(self, num_layers=8, num_features=256, L_pos=6, L_dir=4):
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
        x = self.positional_encoding(x)
        view_direction = self.directional_encoding(view_direction)

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
            if i == len(self.layers) // 2 - 1:
                x = torch.cat([x, view_direction], dim=-1)

        alpha = F.relu(self.alpha_layer(x))
        rgb = torch.sigmoid(self.rgb_layer(x))

        return alpha, rgb
