import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.kan_pinn import pinn
from src.physics import physics
from src.augmented_lagrangian import algl

def train(dataloader: Dataloader, epochs: int = 200, lr_model: float = 1e-3, lr_al: float=1e-3):
    model=pinn()
    phy = physics()
    num_samples=len(dataloader.dataset)
    al_loss_fn=algl(num_samples=num_samples)
    model_params=list(model.parameters()) + list(phy.parameters())
    optimizer_model = optim.AdamW(model_params, lr=lr_model)
    al_params = list(al_loss_fn.parameters())
    optimizer_al=optim.AdamW(al_params,lr=lr_al,maximize=True)
    model.train()
    for epoch in range(epochs):
        epoch_loss=0.0
        for batch_idx, batch in enumerate(dataloader):
            x_features,t,u_true,indices=batch
            t.requires_grad_(True)
            optimizer_model.zero_grad()
            optimizer_al.zero_grad()
            u_hat = model(x_features)
            resphy = phy(t,u_hat)
            batch_lambdas=al_loss_fn.lambdas[indices]
            loss_data=torch.mean((u_hat - u_true) ** 2)
            loss_lambda = torch.mean(batch_lambdas * resphy)
            loss_penalty = (al_loss_fn.sigma/2.0) * torch.mean(resphy ** 2)
            total_loss = loss_data + loss_lambda + loss_penalty
            total_loss.backward()
            optimizer_model.step()
            optimizer_al.step()
            epoch_loss += total_loss.item()

        if (epoch + 1) % 10 == 0:
            print (f"Epoch {epoch + 1}/ {epoch} | Total Loss: {epoch_loss / len(dataloader):.6f} | Sigma: {al_loss_fn.sigma.item():.4f}")
    return model, phy

if __name__ == "__main__":
    pass
