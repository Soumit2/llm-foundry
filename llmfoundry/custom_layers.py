import torch
import torch.nn as nn

# class BandMatrix(nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
        
#         # **Reduce rank to minimize parameters (default: 1/16 instead of 1/4)**
#         self.rank = kwargs.get('rank', max(1, min(in_features, out_features) // 16))
#         self.bandwidth = kwargs.get('bandwidth', min(in_features, out_features) // 8)

#         # **Sparse Representation of W1 and W2**
#         self.sparse_indices = self.create_band_indices()
#         self.W_values = nn.Parameter(torch.randn(len(self.sparse_indices[0])))

#         # Learnable bias
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_features))
#         else:
#             self.register_parameter('bias', None)

#     def create_band_indices(self):
#         """Creates indices for a sparse banded weight matrix."""
#         rows, cols = [], []
#         for i in range(self.out_features):
#             for j in range(max(0, i - self.bandwidth), min(self.in_features, i + self.bandwidth + 1)):
#                 rows.append(i)
#                 cols.append(j)
#         return torch.tensor(rows), torch.tensor(cols)

#     def forward(self, x):
#         """Applies sparse matrix multiplication using indexed values."""
#         weight_matrix = torch.sparse_coo_tensor(
#             indices=torch.stack(self.sparse_indices), 
#             values=self.W_values,
#             size=(self.out_features, self.in_features)
#         ).to_dense()  # Convert back to dense for multiplication
        
#         output = torch.matmul(x, weight_matrix.T)  # Efficient multiplication
#         if self.bias is not None:
#             output += self.bias
#         return output
class BandMatrix(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bandwidth = min(in_features, out_features) // 4
        self.rank = max(1, min(in_features, out_features) // 4)

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
