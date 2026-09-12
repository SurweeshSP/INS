import os
import sys
sys.path.append(r"C:\Users\surwe\Project\INS\model")

import pandas as pd
import numpy as np

def main():
    sm_dir = r"C:\Users\surwe\Project\INS\IO-VNBD\preprocess_data"
    vm_root = r"C:\Users\surwe\Project\INS\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"
    
    # 1. Find all pairs
    s_files = [f for f in os.listdir(sm_dir) if f.startswith("S-") and f.endswith(".csv")]
    
    # Search for V-*.csv files recursively under vm_root
    v_paths = {}
    for dirpath, dirnames, filenames in os.walk(vm_root):
        for f in filenames:
            if f.startswith("V-") and f.endswith(".csv"):
                name = f[2:-4] # e.g. "M", "S1", "Vta1a"
                # Make keys lowercase to avoid case mismatches
                v_paths[name.lower()] = os.path.join(dirpath, f)
                
    pairs = []
    for sf in s_files:
        name = sf[2:-4] # e.g. "M", "S1", "Vta1a"
        key = name.lower()
        # Handle custom mapping if name doesn't match exactly
        if key in v_paths:
            pairs.append({
                'name': name,
                's_path': os.path.join(sm_dir, sf),
                'v_path': v_paths[key]
            })
        else:
            # Try case-insensitive substring match
            matched = False
            for k, p in v_paths.items():
                if k in key or key in k:
                    pairs.append({
                        'name': name,
                        's_path': os.path.join(sm_dir, sf),
                        'v_path': p
                    })
                    matched = True
                    break
            if not matched:
                print(f"Warning: No V-M match found for {sf}")
                
    print(f"Found {len(pairs)} matched S-M and V-M files.")
    
    # 2. Check alignment shifts on a few files to verify
    for i, pair in enumerate(pairs[:5]):
        df_sm = pd.read_csv(pair['s_path'])
        df_vm = pd.read_csv(pair['v_path'], encoding="latin-1")
        df_vm.columns = [c.strip() for c in df_vm.columns]
        
        # Check matching columns
        gyro_pitch = df_sm['GyroPitch'].values * 180.0 / np.pi
        obd_yaw = df_vm['Yaw Rate (deg/sec)'].values
        
        # Let's find best shift using filtered correlation
        def moving_average(a, n=50):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n
            
        gyro_pitch_f = moving_average(gyro_pitch, 50)
        obd_yaw_f = moving_average(obd_yaw, 50)
        
        best_corr = -1
        best_shift = 0
        for shift in range(-100, 100):
            if shift < 0:
                a = gyro_pitch_f[-shift:]
                b = obd_yaw_f[:shift]
            elif shift > 0:
                a = gyro_pitch_f[:-shift]
                b = obd_yaw[shift:] # wait, b needs to be obd_yaw_f
                # let's slice correctly:
                a = gyro_pitch_f[:-shift]
                b = obd_yaw_f[shift:]
            else:
                a = gyro_pitch_f
                b = obd_yaw_f
            corr = abs(np.corrcoef(a, b)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_shift = shift
                
        print(f"Pair {i+1} ({pair['name']}): Best shift = {best_shift} rows, Max Yaw correlation = {best_corr:.4f}")

if __name__ == '__main__':
    main()
