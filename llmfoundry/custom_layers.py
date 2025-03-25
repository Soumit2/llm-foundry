import torch
import torch.nn as nn

class BandMatrix(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Reduce the rank to decrease the parameter count (default: 1/8 of min(in_features, out_features))
        self.rank = kwargs.get('rank', max(1, min(in_features, out_features) // 8))
        self.bandwidth = kwargs.get('bandwidth', min(in_features, out_features) // 4)
        
        # Optional: use sparse initialization to further reduce effective parameter capacity.
        # 'sparsity' is the fraction of weights set to zero (default 50%).
        sparsity = kwargs.get('sparsity', 0.5)
        mask_W1 = (torch.rand(out_features, self.rank) > sparsity).float()
        mask_W2 = (torch.rand(self.rank, in_features) > sparsity).float()
        
        # Learnable weight matrices for low-rank decomposition with sparse initialization
        self.W1 = nn.Parameter(torch.randn(out_features, self.rank) * mask_W1)
        self.W2 = nn.Parameter(torch.randn(self.rank, in_features) * mask_W2)
        
        # Learnable bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Register band mask as a buffer so that it’s part of the state but not learnable.
        self.register_buffer("band_mask", self.create_band_mask())
    
    def create_band_mask(self):
        """Creates a binary mask for the band matrix that enforces a band structure."""
        mask = torch.zeros(self.out_features, self.in_features)
        for i in range(self.out_features):
            for j in range(max(0, i - self.bandwidth), min(self.in_features, i + self.bandwidth + 1)):
                mask[i, j] = 1
        return mask

    def forward(self, x):
        """Applies the band matrix transformation with enforced sparsity."""
        # Compute low-rank product and then apply the band mask
        weight = (self.W1 @ self.W2) * self.band_mask
        output = torch.matmul(x, weight.T)
        if self.bias is not None:
            output += self.bias
        return output
