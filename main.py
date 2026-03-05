import os
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import DataLoader, random_split
from src.data_processor import BatteryDataProcessor
from src.dataset import BatterySOHDataset
from train import train_al_pkan

def load_tri_data(data_path):
    processor=BatteryDataProcessor()
    all_features=[] 
    all_u_true=[]
    all_t_norm = []
    files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    print (f"Found {len(files)} files.")
    for file in files:
        df = pd.read_csv(os.path.join(data_path,file))
        grouped=df.groupby('cycle')
        Q_nominal = df['Qdch'].max()
        for cycle_idx,group in grouped:
            if (len(group) < 10): continue
            v_raw = group['V'].values
            t_raw = group['T'].values
            time_raw = np.arrange(len(v_raw))
            v_scaled, t_scaled = processor.interpolate_and_scale(time_raw,v_raw,t_raw)
            t_norm = cycle_idx/2000.0
            dch_time=len(group)/100.0
            features=processor.build_feature_vector(v_scaled,t_scaled,dch_time,t_norm)
            u_true = (group['Qdch'].mean()/Q_nominal)
            all_features.append(features)
            all_u_true.append(u_true)
            all_t_norm.append(t_norm)

        return np.array(all_features), np.array(all_u_true), np.array(all_t_norm)
    if __name__ == "__main__":
        raw_data_path="data/raw"
        X,U,T = load_tri_data(raw_data_path)
        dataset=BatterySOHDataset(X,U,T)
        train_size=int(0.8 * len(dataset))
        test_size= len(dataset) - train_size
        train_dataset,test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)
        model,physics=train_al_pkan(train_loader,epochs=100)
        torch.save(model.state_dict(),"kan_pinn_model.pth")
        
