import sys
import os
sys.path.append(r"C:\Users\surwe\Project\INS\model")

import pandas as pd
from preprocess import find_optimal_shift, create_drive_windows

def main():
    sm_path = r"C:\Users\surwe\Project\INS\IO-VNBD\preprocess_data\S-M.csv"
    vm_path = r"C:\Users\surwe\Project\INS\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset\M (Driver B)\V-M.csv"
    
    df_sm = pd.read_csv(sm_path)
    df_vm = pd.read_csv(vm_path, encoding="latin-1")
    df_vm.columns = [c.strip() for c in df_vm.columns]
    
    shift = find_optimal_shift(df_sm, df_vm)
    print(f"Optimal shift: {shift}")
    
    X_w, y_s_w, y_e_w = create_drive_windows(df_sm, df_vm, shift, window_len=40, stride=10)
    
    print(f"Generated {len(X_w)} windows.")
    print("Audit Target Alignment (First 5 windows):")
    for i in range(5):
        start_idx = i * 10
        end_idx = i * 10 + 39
        
        if shift > 0:
            vm_aligned = df_vm.iloc[shift:].reset_index(drop=True)
        elif shift < 0:
            vm_aligned = df_vm.iloc[:shift].reset_index(drop=True)
        else:
            vm_aligned = df_vm.reset_index(drop=True)
            
        speed_col = [c for c in vm_aligned.columns if 'Velocity' in c][0]
        actual_end_speed = vm_aligned[speed_col].values[end_idx] / 3.6
        actual_start_speed = vm_aligned[speed_col].values[start_idx] / 3.6
        
        print(f"Window {i}: start_idx={start_idx} (speed {actual_start_speed:.4f}), end_idx={end_idx} (speed {actual_end_speed:.4f}) -> target y={y_s_w[i]:.4f}")
        assert abs(actual_end_speed - y_s_w[i]) < 1e-5, "Target mismatch!"

if __name__ == '__main__':
    main()
