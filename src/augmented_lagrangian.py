import torch 
import torch.nn as nn

class algloss(nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.lambdas = nn.Parameter(torch.zeros(num_samples,1),requires_grad=True)
        self.sigma=nn.Parameter(torch.tensor(1.0), requires_grad=True)


        def forward(self, u_hat, u_true, resphy):
            ld = torch.mean((u_hat-u_true) ** 2)
            ll = torch.mean(self.lambdas * resphy)
            lp = (self.sigma / 2.0) * torch.mean(resphy ** 2)
            tl = ld + ll + lp

            return tl, ld, lp


