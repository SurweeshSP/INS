import os
import pandas as pd
import numpy as np

def main():
    sm_prep_path = r"C:\Users\surwe\Project\INS\IO-VNBD\preprocess_data\S-M.csv"
    vm_raw_path = r"C:\Users\surwe\Project\INS\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset\M (Driver B)\V-M.csv"
    
    df_sm = pd.read_csv(sm_prep_path)
    df_vm = pd.read_csv(vm_raw_path, encoding="latin-1")
    df_vm.columns = [c.strip() for c in df_vm.columns]
    
    # Check correlation of preprocessed features with vehicle dynamics
    # Is there a shift?
    # Let's compare speed targets or yaw rates
    # Preprocessed GyroYaw vs OBD Yaw Rate (GyroYaw is in rad/sec, OBD in deg/sec)
    gyro_yaw = df_sm['GyroYaw'].values * 180.0 / np.pi
    obd_yaw = df_vm['Yaw Rate (deg/sec)'].values
    
    # Direct correlation
    direct_corr = np.corrcoef(gyro_yaw, obd_yaw)[0, 1]
    print("Direct row-by-row correlation of GyroYaw with OBD Yaw Rate:", direct_corr)
    
    # We saw in audit_axes.py that GyroPitch had 0.3648 correlation, not GyroYaw.
    # Let's check all gyro axes in preprocessed S-M
    gyro_pitch = df_sm['GyroPitch'].values * 180.0 / np.pi
    gyro_roll = df_sm['GyroRoll'].values * 180.0 / np.pi
    
    print("\nPreprocessed Gyro axes correlation with OBD Yaw Rate:")
    print(f"  GyroYaw:   {np.corrcoef(gyro_yaw, obd_yaw)[0, 1]:.4f}")
    print(f"  GyroPitch: {np.corrcoef(gyro_pitch, obd_yaw)[0, 1]:.4f}")
    print(f"  GyroRoll:  {np.corrcoef(gyro_roll, obd_yaw)[0, 1]:.4f}")
    
    # Let's check if there is an optimal shift between preprocessed S-M and V-M!
    # Let's find shift by maximizing correlation of GyroPitch with OBD Yaw
    best_corr = -1
    best_shift = 0
    for shift in range(-50, 50):
        if shift < 0:
            a = gyro_pitch[-shift:]
            b = obd_yaw[:shift]
        elif shift > 0:
            a = gyro_pitch[:-shift]
            b = obd_yaw[shift:]
        else:
            a = gyro_pitch
            b = obd_yaw
        corr = abs(np.corrcoef(a, b)[0, 1])
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
            
    print(f"\nBest row shift: {best_shift} rows, Max correlation: {best_corr:.4f}")

if __name__ == '__main__':
    main()
