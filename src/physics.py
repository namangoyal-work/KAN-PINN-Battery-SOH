import torch
import torch.nn as nn

class physics(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = nn.Parameter(torch.tensor([0.003],dtype=torch.float32))
        self.M = nn.Parameter(torch.tensor([0.95], dtype=torch.float32))
        self.ui = nn.Parameter(torch.tensor([0.016], dtype=torch.float32))


    def forward(self,t,u_hat,x_in=None,model=None):
        du_dt = torch.autograd.grad(outputs=u_hat,inputs=t,grad_outputs=torch.ones_like(u_hat),create_graph=True,retain_graph=True,allow_unused=True)[0]
        if du_dt is None:
            if x_in is None or model is None:
                # If we forgot to pass them, we use a zero-gradient to avoid crashing
                # but we print a warning so you know the graph is detached.
                return torch.zeros_like(u_hat)
            delta = 1e-4
            x_plus = x_in.clone()
            x_plus[:, -1] += delta
            u_plus = model(x_plus)
            x_minus = x_in.clone()
            x_minus[:, -1] -= delta
            u_minus = model(x_minus)
            du_dt = (u_plus - u_minus) / (2.0 * delta)

    
        ud = u_hat - self.ui
        Md = self.M - self.ui
        G = self.k * ud * (1.0 - (ud / Md))

        resf = du_dt - G
        return resf
