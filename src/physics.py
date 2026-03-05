import torch
import torch.nn as nn

class physics(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = nn.Parameter(torch.tensor([0.003],dtype=torch.float32))
        self.M = nn.Parameter(torch.tensor([0.95], dtype=torch.float32))
        self.ui = nn.Parameter(torch.tensor([0.016], dtype=torch.float32))


    def forward(self,t,u_hat):
        du_dt = torch.autograd.grad(outputs=u_hat,inputs=t,grad_outputs=torch.ones_like(u_hat),create_graph=True,retain_graph=True)[0]
        ud = u_hat - self.ui
        Md = self.M - self.ui
        G = self.k * ud * (1.0 - (ud/Md))
        resf = du_dt - G
        return resf
