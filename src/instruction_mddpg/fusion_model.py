import torch
from torch import nn

class GatedAttention(nn.Module):
    def __init__(self, dim_x, dim_y, hidden_dim):
        super().__init__()
        """
        dim_x: width (or height) dim of conv layer
        dim_y: sentence embedding dims
        hidden_dim: linear embedding
        Ex: (12, 256, 64) # (dim_x, dim_y, hidden_dim)
        """
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.hidden_dim = hidden_dim
        # Gated-Attention layers
        self.attn_linear = nn.Linear(dim_y, hidden_dim)
        
    def forward(self, x, y):
        """
        x: conv maps
        y: sentence embedding
        """
        y_attention = torch.sigmoid(self.attn_linear(y))
        y_attention = y_attention.unsqueeze(2).unsqueeze(3)
        y_attention = y_attention.expand(1, self.hidden_dim, self.dim_x, self.dim_x)

        z = x*y_attention
        z = z.view(z.size(0), -1)

        return z