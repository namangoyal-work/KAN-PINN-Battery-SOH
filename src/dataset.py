import torch 
from torch.utils.data import Dataset

class BatterySOHDataset(Dataset):
    def __init__(self, feature_matrix, true_soh, cycle_indices):
        self.X = torch.tensor(feature_matrix, dtype=torch.float32)
        closs = 1.0-true_soh
        self.U_true = torch.tensor(closs,dtype=torch.float32).unsqueeze(1)
        self.t=torch.tensor(cycle_indices,dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return self.X[idx], self.t[idx],self.U_true[idx],idx

