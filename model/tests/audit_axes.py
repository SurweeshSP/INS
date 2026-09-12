import os
import sys
sys.path.append(r"C:\Users\surwe\Project\INS\model")

import pandas as pd
import numpy as np

def moving_average(a, n=50):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():
    sm_raw_path = r"C:\Users\surwe\Project\INS\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset\M (Driver B)\S-M.csv"
    vm_raw_path = r"C:\Users\surwe\Project\INS\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset\M (Driver B)\V-M.csv"
    
    df_sm = pd.read_csv(sm_raw_path, encoding="latin-1")
    df_vm = pd.read_csv(vm_raw_path, encoding="latin-1")
    
    df_sm.columns = [c.strip() for c in df_sm.columns]
    df_vm.columns = [c.strip() for c in df_vm.columns]
    
    # Parse times
    def parse_sm_time(date_str):
        time_part = date_str.split(' ')[1]
        parts = time_part.split(':')
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        ms = int(parts[3]) if len(parts) > 3 else 0
        return h * 3600 + m * 60 + s + ms / 1000.0
        
    sm_times = df_sm['DATE (YYYY-MO-DD HH-MI-SS_SSS)'].apply(parse_sm_time).values
    vm_times = df_vm['Time Since Start of Day (seconds)'].values
    
    offset = 3604.0
    
    # Get column names robustly
    acc_x_col = [c for c in df_sm.columns if 'ACCELEROMETER X' in c][0]
    acc_y_col = [c for c in df_sm.columns if 'ACCELEROMETER Y' in c][0]
    acc_z_col = [c for c in df_sm.columns if 'ACCELEROMETER Z' in c][0]
    
    grav_x_col = [c for c in df_sm.columns if 'GRAVITY X' in c][0]
    grav_y_col = [c for c in df_sm.columns if 'GRAVITY Y' in c][0]
    grav_z_col = [c for c in df_sm.columns if 'GRAVITY Z' in c][0]
    
    gyro_yaw_col = [c for c in df_sm.columns if 'GYROSCOPE Yaw' in c][0]
    gyro_pitch_col = [c for c in df_sm.columns if 'GYROSCOPE Pitch' in c][0]
    gyro_roll_col = [c for c in df_sm.columns if 'GYROSCOPE Roll' in c][0]
    
    # Get gyroscope raw signals
    gyro_yaw = df_sm[gyro_yaw_col].values * 180.0 / np.pi
    gyro_pitch = df_sm[gyro_pitch_col].values * 180.0 / np.pi
    gyro_roll = df_sm[gyro_roll_col].values * 180.0 / np.pi
    
    # OBD yaw rate
    obd_yaw = df_vm['Yaw Rate (deg/sec)'].values
    obd_yaw_aligned = np.interp(sm_times - offset, vm_times, obd_yaw)
    
    # Apply moving average filter to reduce high frequency noise
    w_size = 50
    gyro_yaw_f = moving_average(gyro_yaw, w_size)
    gyro_pitch_f = moving_average(gyro_pitch, w_size)
    gyro_roll_f = moving_average(gyro_roll, w_size)
    obd_yaw_f = moving_average(obd_yaw_aligned, w_size)
    
    print("Filtered Correlation with OBD Yaw Rate:")
    print(f"  Gyro Yaw:   {np.corrcoef(gyro_yaw_f, obd_yaw_f)[0, 1]:.4f}")
    print(f"  Gyro Pitch: {np.corrcoef(gyro_pitch_f, obd_yaw_f)[0, 1]:.4f}")
    print(f"  Gyro Roll:  {np.corrcoef(gyro_roll_f, obd_yaw_f)[0, 1]:.4f}")
    print(f"  -Gyro Yaw:  {np.corrcoef(-gyro_yaw_f, obd_yaw_f)[0, 1]:.4f}")
    print(f"  -Gyro Pitch:{np.corrcoef(-gyro_pitch_f, obd_yaw_f)[0, 1]:.4f}")
    print(f"  -Gyro Roll: {np.corrcoef(-gyro_roll_f, obd_yaw_f)[0, 1]:.4f}")
    
    # Let's check accelerometer alignment
    obd_acc_long = df_vm['Indicated Longitudinal Acceleration (g)'].values * 9.80665
    obd_acc_long_aligned = np.interp(sm_times - offset, vm_times, obd_acc_long)
    
    # Smartphone accelerations (linear)
    acc_x = df_sm[acc_x_col].values - df_sm[grav_x_col].values
    acc_y = df_sm[acc_y_col].values - df_sm[grav_y_col].values
    acc_z = df_sm[acc_z_col].values - df_sm[grav_z_col].values
    
    # Filter accelerations
    acc_x_f = moving_average(acc_x, w_size)
    acc_y_f = moving_average(acc_y, w_size)
    acc_z_f = moving_average(acc_z, w_size)
    obd_acc_long_f = moving_average(obd_acc_long_aligned, w_size)
    
    print("\nFiltered Correlation with OBD Longitudinal Acceleration:")
    print(f"  Linear Acc X:  {np.corrcoef(acc_x_f, obd_acc_long_f)[0, 1]:.4f}")
    print(f"  Linear Acc Y:  {np.corrcoef(acc_y_f, obd_acc_long_f)[0, 1]:.4f}")
    print(f"  Linear Acc Z:  {np.corrcoef(acc_z_f, obd_acc_long_f)[0, 1]:.4f}")
    print(f"  -Linear Acc X: {np.corrcoef(-acc_x_f, obd_acc_long_f)[0, 1]:.4f}")
    print(f"  -Linear Acc Y: {np.corrcoef(-acc_y_f, obd_acc_long_f)[0, 1]:.4f}")
    print(f"  -Linear Acc Z: {np.corrcoef(-acc_z_f, obd_acc_long_f)[0, 1]:.4f}")

if __name__ == '__main__':
    main()
