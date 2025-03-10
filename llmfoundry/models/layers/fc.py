# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from llmfoundry.layers_registry import fcs

class BandMatrix(nn.Module):
    """A custom Band Matrix layer that retains only a banded structure."""

    def __init__(self, in_features: int, out_features: int, bandwidth: int):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bandwidth (int): Maximum number of diagonals to retain.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bandwidth = bandwidth

        # Learnable weight matrix
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        # Mask to enforce banded structure
        self.register_buffer("band_mask", self.create_band_mask())

    def create_band_mask(self):
        """Creates a binary mask to enforce the band matrix structure."""
        mask = torch.zeros(self.out_features, self.in_features)
        for i in range(self.out_features):
            for j in range(max(0, i - self.bandwidth), min(self.in_features, i + self.bandwidth + 1)):
                mask[i, j] = 1
        return mask

    def forward(self, x):
        """
        Applies the band matrix transformation.
        Args:
            x (Tensor): Input tensor of shape (batch, in_features).
        Returns:
            Tensor: Transformed output of shape (batch, out_features).
        """
        return torch.matmul(x, (self.weight * self.band_mask).T)

# Register the BandMatrix layer in fcs
fcs.register(
    "BandMatrix",
    BandMatrix,
    in_features=int,
    out_features=int,
    bandwidth=int,
)
