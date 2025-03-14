import torch
import torch.nn as nn

class ToeplitzMatrix(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable diagonal elements (parameterizing only first row)
        self.first_row = nn.Parameter(torch.randn(out_features))
        self.first_col = nn.Parameter(torch.randn(in_features))

        # Learnable bias
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter("bias", None)

    def create_toeplitz_matrix(self):
        matrix = torch.zeros(self.out_features, self.in_features)
        for i in range(self.out_features):
            for j in range(self.in_features):
                if i >= j:
                    matrix[i, j] = self.first_col[i - j]
                else:
                    matrix[i, j] = self.first_row[j - i]
        return matrix

    def forward(self, x):

        toeplitz_matrix = self.create_toeplitz_matrix()
        output = torch.matmul(x, toeplitz_matrix.T)
        if self.bias is not None:
            output += self.bias
        return output
