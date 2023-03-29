import torch
import torch.nn as nn
from torch.nn import functional as F


class ResNet(nn.Module):
    """
    simple resnet with fully connected layers
    and ReLU activations

    if dim_list is (n,...,m) then the network will take in tensors with
    'n' features and outputs 'm' values
    """

    def __init__(self, dim_list: list[int]):
        super().__init__()
        in_dim, dim_list[0] = dim_list[0], 0
        self.linears = nn.ModuleList(
            [nn.Linear(a + in_dim, b) for (a, b) in zip(dim_list[:-1], dim_list[1:])]
        )

    def forward(self, x):
        y = self.linears[0](x)
        y = F.relu(y)
        for layer in self.linears[1:-1]:
            z = torch.hstack((x, y))
            y = layer(z)
            y = F.relu(y)
        z = torch.hstack((x, y))
        y = self.linears[-1](z)
        return y
