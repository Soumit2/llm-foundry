# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from llmfoundry.layers_registry import fcs

fcs.register('torch', func=nn.Linear)

try:
    import transformer_engine.pytorch as te
    fcs.register('te', func=te.Linear)
except:
    pass

class BandLinear(nn.Module):
    def __init__(self, in_features, out_features, bandwidth=1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bandwidth = bandwidth
        self.bias = bias

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.register_buffer("mask", self.create_band_mask())

    def create_band_mask(self):
        """Creates a binary mask for the band matrix structure."""
        mask = torch.zeros(self.out_features, self.in_features)
        for i in range(self.out_features):
            for j in range(self.in_features):
                if abs(i - j) <= self.bandwidth:  # Keep elements in the band
                    mask[i, j] = 1
        return mask

    def forward(self, x):
        masked_weight = self.weight * self.mask  # Enforce band structure
        output = x @ masked_weight.T  # Standard matrix multiplication
        if self.bias is not None:
            output += self.bias
        return output

fcs.register('BandLinear', func=BandLinear)
