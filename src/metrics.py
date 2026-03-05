import torch

def calculate_metrics(u_pred, u_true):
    with torch.no_grad():
        mae = torch.mean(torch.abs(u_pred - u_true)).item() * 100.0  
        epsilon = 1e-8
        rmspe = torch.sqrt(torch.mean(((u_pred - u_true) / (u_true + epsilon)) ** 2)).item() * 100.0
        ss_res = torch.sum((u_true - u_pred) ** 2)
        ss_tot = torch.sum((u_true - torch.mean(u_true)) ** 2)
        r2 = (1 - (ss_res / (ss_tot + epsilon))).item()
        
    return {"RMSPE": rmspe, "MAE": mae, "R2": r2}
