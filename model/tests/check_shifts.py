import os
import pandas as pd
import numpy as np

def main():
    sm_prep_path = r"C:\Users\surwe\Project\INS\IO-VNBD\preprocess_data\S-M.csv"
    vm_raw_path = r"C:\Users\surwe\Project\INS\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset\M (Driver B)\V-M.csv"
    
    df_sm = pd.read_csv(sm_prep_path)
    df_vm = pd.read_csv(vm_raw_path, encoding="latin-1")
    df_vm.columns = [c.strip() for c in df_vm.columns]
    
    gyro_pitch = df_sm['GyroPitch'].values * 180.0 / np.pi
    obd_yaw = df_vm['Yaw Rate (deg/sec)'].values
    
    # We can also compute moving average filter to get cleaner correlation
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
            b = obd_yaw_f[shift:]
        else:
            a = gyro_pitch_f
            b = obd_yaw_f
        corr = np.corrcoef(a, b)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
            
    print(f"Filtered Yaw Rate correlation: Best shift = {best_shift} rows, Max correlation = {best_corr:.4f}")
    
    # Check correlation at shift = 29
    if 29 < len(gyro_pitch_f):
        corr_29 = np.corrcoef(gyro_pitch_f[:-29], obd_yaw_f[29:])[0, 1]
        print(f"Correlation at shift = 29 rows: {corr_29:.4f}")
        
    # Check correlation at shift = 13
    if 13 < len(gyro_pitch_f):
        corr_13 = np.corrcoef(gyro_pitch_f[:-13], obd_yaw_f[13:])[0, 1]
        print(f"Correlation at shift = 13 rows: {corr_13:.4f}")

if __name__ == '__main__':
    main()
