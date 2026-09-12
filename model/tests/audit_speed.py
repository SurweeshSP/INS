import os
import sys
# Set path to include c:\Users\surwe\Project\INS\model
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
    
    # Parse times
    def parse_sm_time(date_str):
        time_part = date_str.split(' ')[1]
        parts = time_part.split(':')
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        ms = int(parts[3]) if len(parts) > 3 else 0
        return h * 3600 + m * 60 + s + ms / 1000.0
        
    sm_times = df_sm['DATE (YYYY-MO-DD HH-MI-SS_SSS)'].apply(parse_sm_time).values
    vm_times = df_vm['Time Since Start of Day (seconds)'].values
    
    # Optimal offset
    offset = 3604.0
    
    # Align speeds
    sm_speed = df_sm['GPS SPEED (Kmh)'].values
    vm_speed = df_vm['Velocity (km/hr)'].values
    vm_speed_aligned = np.interp(sm_times - offset, vm_times, vm_speed)
    
    # Print sample predictions vs actual speeds to show alignment
    # Let's select indices where speed changes
    print("Aligned vs Unaligned Speed Correlation:")
    print("  Direct Row-by-Row Correlation:", np.corrcoef(sm_speed, vm_speed)[0, 1])
    print("  Aligned Time-based Correlation:", np.corrcoef(sm_speed, vm_speed_aligned)[0, 1])
    
    # 2. Check alignment of physical IMU signals:
    # Smartphone Gyroscope Yaw vs OBD Yaw Rate (OBD is in deg/sec, Gyro is in rad/sec)
    gyro_yaw = df_sm['GYROSCOPE Yaw (rad/s)'].values
    obd_yaw = df_vm['Yaw Rate (deg/sec)'].values
    
    # Convert gyro to deg/sec
    gyro_yaw_deg = gyro_yaw * 180.0 / np.pi
    
    # Interpolate
    obd_yaw_aligned = np.interp(sm_times - offset, vm_times, obd_yaw)
    
    print("\nAligned vs Unaligned Yaw Rate Correlation:")
    print("  Direct Row-by-Row Correlation:", np.corrcoef(gyro_yaw_deg, obd_yaw)[0, 1])
    print("  Aligned Time-based Correlation:", np.corrcoef(gyro_yaw_deg, obd_yaw_aligned)[0, 1])
    
    # Let's print a few samples
    print("\nFirst 10 aligned samples of Yaw Rate (deg/s):")
    print("  SM Gyro (deg/s) | OBD Yaw Rate (deg/s)")
    for i in range(100, 110):
        print(f"  {gyro_yaw_deg[i]:14.4f} | {obd_yaw_aligned[i]:14.4f}")

if __name__ == '__main__':
    main()
