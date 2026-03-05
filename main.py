import os
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader, random_split

print("!!! SCRIPT STARTED !!!", flush=True)

from src.data_processor import BatteryDataProcessor
from src.dataset import BatterySOHDataset
from train import train_al_pkan

quicktest = True;
def load_tri_mat_data(data_path):
    print(f"Checking directory: {os.path.abspath(data_path)}", flush=True)
    processor = BatteryDataProcessor()
    all_features, all_u_true, all_t_norm = [], [], []
    
    files = [f for f in os.listdir(data_path) if f.endswith('.mat')]
    if not files:
        print("No .mat files found!", flush=True)
        return None

    for file in files:
        file_path = os.path.join(data_path, file)
        print(f"\nOpening HDF5 file: {file}...", flush=True)
        
        with h5py.File(file_path, 'r') as f:
            # In MATLAB v7.3, batch is a Group. Its fields are Datasets of references.
            batch_group = f['batch']
            
            # Get the references for the 'cycles' struct for all cells
            cycles_field = batch_group['cycles']
            
            # The shape is usually (num_cells, 1) or (1, num_cells)
            num_cells = cycles_field.shape[0] if cycles_field.shape[0] > 1 else cycles_field.shape[1]
            print(f"Successfully identified {num_cells} battery cells.", flush=True)

            for i in range(num_cells):
                if quicktest and i >= 2:
                    break

                try:
                    # Dereference the cycles struct for cell 'i'
                    cycles_ref = cycles_field[i, 0] if cycles_field.shape[0] > 1 else cycles_field[0, i]
                    cell_cycles = f[cycles_ref] # This is a Group containing V, T, Qd, etc.
                    
                    if i == 0:
                        print(f"Fields available in cycles: {list(cell_cycles.keys())}", flush=True)
                    
                    # The original MIT dataset uses 'Qd' for discharge capacity
                    Qd_field_name = 'Qd' if 'Qd' in cell_cycles.keys() else 'Qdch'
                    
                    V_refs = cell_cycles['V']
                    T_refs = cell_cycles['T']
                    Qd_refs = cell_cycles[Qd_field_name]
                    
                    num_cycles = V_refs.shape[0] if V_refs.shape[0] > 1 else V_refs.shape[1]
                    
                    # Get Nominal Capacity from cycle 2 (cycle 0 and 1 can be noisy)
                    q_nom_ref = Qd_refs[2, 0] if Qd_refs.shape[0] > 1 else Qd_refs[0, 2]
                    q_nominal = np.max(np.array(f[q_nom_ref]).flatten())
                    if np.isnan(q_nominal) or q_nominal < 0.5:
                        print(f"Skipping cell {i} due to corrupted q_nominal: {q_nominal}", flush=True)
                        continue

                    for j in range(1, num_cycles - 1):
                        try:
                            # Dereference V, T, and Qd for cycle 'j'
                            v_ref = V_refs[j, 0] if V_refs.shape[0] > 1 else V_refs[0, j]
                            t_ref = T_refs[j, 0] if T_refs.shape[0] > 1 else T_refs[0, j]
                            qd_ref = Qd_refs[j, 0] if Qd_refs.shape[0] > 1 else Qd_refs[0, j]
                            
                            V_raw = np.array(f[v_ref]).flatten()
                            T_raw = np.array(f[t_ref]).flatten()
                            Qd = np.max(np.array(f[qd_ref]).flatten())
                            
                            if len(V_raw) < 10: continue

                            # Interpolation
                            v_scaled, t_scaled = processor.interpolate_and_scale(
                                np.arange(len(V_raw)), V_raw, T_raw
                            )
                            
                            # Build the 2050-dimensional input vector
                            t_norm = j / 2000.0
                            dch_time = len(V_raw) / 100.0
                            features = processor.build_feature_vector(v_scaled, t_scaled, dch_time, t_norm)
                            
                            all_features.append(features)
                            all_u_true.append(1.0 - (Qd / q_nominal))
                            all_t_norm.append(t_norm)
                            
                        except Exception:
                            continue
                            
                    if (i + 1) % 10 == 0:
                        print(f"Extracted {i+1}/{num_cells} cells. Total cycles: {len(all_features)}", flush=True)

                except Exception as e:
                    print(f"Skipping cell {i} due to internal structure error.", flush=True)
                    continue

    return np.array(all_features), np.array(all_u_true), np.array(all_t_norm)

# --- EXECUTION ---
data_path = "data/raw"
result = load_tri_mat_data(data_path)

if result is not None and len(result[0]) > 0:
    X, U, T = result
    print(f"\nEXTRACTION SUCCESS. Total cycles prepared for training: {len(X)}", flush=True)
    
    dataset = BatterySOHDataset(X, U, T)
    train_ds, test_ds = random_split(dataset, [0.8, 0.2]) 
    
    
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    print("\n==================================================")
    print("STARTING AL-PKAN TRAINING")
    print("==================================================")
    model, physics = train_al_pkan(loader, epochs=2)
    
    torch.save(model.state_dict(), "kan_pinn_model.pth")
    print("FINISHED. Model saved as kan_pinn_model.pth.", flush=True)
else:
    print("\nFAILED: Data extraction yielded 0 cycles.", flush=True)
