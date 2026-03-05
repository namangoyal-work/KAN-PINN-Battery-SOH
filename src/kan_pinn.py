import torch 
import torch.nn as nn
from kan import KAN

class pinn(nn.Module):
    def __init__(self, input_dim=2050, hidden_dim=64, grid_size=2, spline_order=3):
        super().__init__()
        self.kan = KAN(
            width=[input_dim, hidden_dim, hidden_dim, 1],
            grid=grid_size,
            k=spline_order
        )

    def forward(self, x):
        
        u_hat = self.kan(x.to(torch.float32))
        return u_hat
