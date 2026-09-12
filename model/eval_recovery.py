import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.model_factory import get_model
from training.train_two_models import get_drive_pairs, split_pairs
from preprocess_vehicle import create_vehicle_windows
from preprocess import find_optimal_shift
from models.evaluation import evaluate_ins_ukf_trajectory

def evaluate_on_full_drives_local(model, pairs, mean, std, device):
    model.eval()
    speed_preds_all, speed_trues_all = [], []
    ax_vehicle_all, ml_speeds_all, ml_vars_all, ml_confs_all = [], [], [], []
    
    for pair in pairs:
        try:
            df_sm = pd.read_csv(pair['s_path'])
            df_vm = pd.read_csv(pair['v_path'], encoding="latin-1")
            df_vm.columns = [c.strip() for c in df_vm.columns]
            
            sm_rename = {'GYROSCOPE Pitch (rad/s)': 'GyroPitch'}
            for col in df_sm.columns:
                for k, v in sm_rename.items():
                    if k in col:
                        df_sm.rename(columns={col: v}, inplace=True)
                        
            shift = find_optimal_shift(df_sm, df_vm)
            X_w, y_s_w, y_e_w = create_vehicle_windows(df_sm, df_vm, shift, window_len=40, stride=10)
            if len(X_w) == 0:
                continue
                
            X_w_norm = (np.array(X_w, dtype=np.float32) - mean.reshape(1, 1, 6)) / std.reshape(1, 1, 6)
            X_tensor = torch.tensor(X_w_norm, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                out = model(X_tensor)
                preds = out["speed_mps"].cpu().numpy()
                if "motion_class" in out:
                    classes = out["motion_class"].cpu().numpy()
                else:
                    classes = np.zeros_like(preds)
                vars_pred = out["speed_variance"].cpu().numpy()
                confs_pred = out["confidence"].cpu().numpy()
                
            speed_preds_all.extend(preds)
            speed_trues_all.extend(y_s_w)
            
            ax_vals = [w[-1, 0] for w in X_w]
            ax_vehicle_all.extend(ax_vals)
            ml_speeds_all.extend(preds)
            ml_vars_all.extend(vars_pred)
            ml_confs_all.extend(confs_pred)
        except Exception as e:
            print(f"Error {pair['name']}: {e}")
            
    speed_preds_all = np.array(speed_preds_all)
    speed_trues_all = np.array(speed_trues_all)
    
    mae = mean_absolute_error(speed_trues_all, speed_preds_all)
    r2 = r2_score(speed_trues_all, speed_preds_all)
    
    drift60 = evaluate_ins_ukf_trajectory(ax_vehicle_all, speed_trues_all, ml_speeds_all, ml_vars_all, ml_confs_all, dt=0.1, outage_duration=60)
    
    return {"mae": mae, "r2": r2, "drift60": drift60}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = r"C:\Users\surwe\Project\INS\model\preprocess_data\vehicle_frame"
    
    mean = np.load(os.path.join(out_dir, "mean.npy"))
    std = np.load(os.path.join(out_dir, "std.npy"))
    
    model = get_model(model_name="Model4", input_dim=6).to(device)
    ckpt_path = os.path.join("checkpoints", "best_accuracy_idr_recovery.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    pairs = get_drive_pairs()
    _, _, test_pairs = split_pairs(pairs)
    
    results = evaluate_on_full_drives_local(model, test_pairs, mean, std, device)
    
    drift = results['drift60']
    
    print(f"Speed MAE: {results['mae']:.2f} m/s")
    print(f"Speed R2: {results['r2']:.4f}")
    print(f"Final 60s Drift: {drift:.2f}%")
    
    if drift <= 11.68:
        print("RESULT: SUCCESS. Golden baseline recovered.")
    else:
        print("RESULT: REGRESSION")

if __name__ == "__main__":
    main()
