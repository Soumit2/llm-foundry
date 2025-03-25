import torch
import torch.nn as nn

class BandMatrix(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bandwidth = kwargs.get('bandwidth', min(in_features, out_features) // 4)
        self.rank = kwargs.get('rank', max(1, min(in_features, out_features) // 4))

        # Learnable weight matrices for low-rank decomposition
        self.W1 = nn.Parameter(torch.randn(out_features, self.rank))
        self.W2 = nn.Parameter(torch.randn(self.rank, in_features))

        # Learnable bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Register band mask as a buffer
        self.register_buffer("band_mask", self.create_band_mask())

    def create_band_mask(self):
        """Creates a binary mask for the band matrix."""
        mask = torch.zeros(self.out_features, self.in_features)
        for i in range(self.out_features):
            for j in range(max(0, i - self.bandwidth), min(self.in_features, i + self.bandwidth + 1)):
                mask[i, j] = 1
        return mask

    def forward(self, x):
        """Applies the band matrix transformation."""
        weight = (self.W1 @ self.W2) * self.band_mask  # Enforce band structure
        output = torch.matmul(x, weight.T)  # Matrix multiplication
        if self.bias is not None:
            output += self.bias  # Add bias if present
        return output
