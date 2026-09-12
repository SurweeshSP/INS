import os
import sys
sys.path.append(r"C:\Users\surwe\Project\INS\model")

import pandas as pd
import numpy as np

def main():
    sm_raw_path = r"C:\Users\surwe\Project\INS\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset\M (Driver B)\S-M.csv"
    vm_raw_path = r"C:\Users\surwe\Project\INS\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset\M (Driver B)\V-M.csv"
    
    df_sm = pd.read_csv(sm_raw_path, encoding="latin-1")
    df_vm = pd.read_csv(vm_raw_path, encoding="latin-1")
    
    df_sm.columns = [c.strip() for c in df_sm.columns]
    df_vm.columns = [c.strip() for c in df_vm.columns]
    
    # 1. Parse timestamps correctly
    def parse_sm_time(date_str):
        # Format: '2019-09-07 09:13:29:506'
        time_part = date_str.split(' ')[1]
        parts = time_part.split(':')
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        ms = int(parts[3]) if len(parts) > 3 else 0
        return h * 3600 + m * 60 + s + ms / 1000.0
        
    sm_times = df_sm['DATE (YYYY-MO-DD HH-MI-SS_SSS)'].apply(parse_sm_time).values
    vm_times = df_vm['Time Since Start of Day (seconds)'].values
    
    print("S-M time start:", sm_times[0], "end:", sm_times[-1])
    print("V-M time start:", vm_times[0], "end:", vm_times[-1])
    
    # Let's check the clock offset using GPS Speed correlation
    sm_speed = df_sm['GPS SPEED (Kmh)'].values
    vm_speed = df_vm['Velocity (km/hr)'].values
    
    # Let's find best shift using correlation
    best_corr = -1
    best_offset = 0
    
    # Grid search offset around 3601.1
    for offset in np.linspace(3600.0, 3604.0, 4000):
        # Interpolate vm_speed onto sm_times - offset
        vm_speed_interp = np.interp(sm_times - offset, vm_times, vm_speed)
        corr = np.corrcoef(sm_speed, vm_speed_interp)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_offset = offset
            
    print("Best offset (seconds):", best_offset)
    print("Max correlation with offset:", best_corr)
    
    # Let's check the correlation with direct row-by-row mapping
    direct_corr = np.corrcoef(sm_speed, vm_speed)[0, 1]
    print("Direct row-by-row correlation:", direct_corr)
    
    # Let's check if the lag changes over time (clock drift)
    n_splits = 5
    seg_len = len(sm_times) // n_splits
    for i in range(n_splits):
        idx_start = i * seg_len
        idx_end = (i + 1) * seg_len
        seg_sm_times = sm_times[idx_start:idx_end]
        seg_sm_speed = sm_speed[idx_start:idx_end]
        
        seg_best_corr = -1
        seg_best_offset = 0
        for offset in np.linspace(3600.0, 3604.0, 400):
            vm_speed_interp = np.interp(seg_sm_times - offset, vm_times, vm_speed)
            corr = np.corrcoef(seg_sm_speed, vm_speed_interp)[0, 1]
            if corr > seg_best_corr:
                seg_best_corr = corr
                seg_best_offset = offset
        print(f"Segment {i+1}: Best offset = {seg_best_offset:.3f} s, Correlation = {seg_best_corr:.4f}")

if __name__ == '__main__':
    main()
