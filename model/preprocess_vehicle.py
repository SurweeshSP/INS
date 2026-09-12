import os
import pandas as pd
import numpy as np
from preprocess import moving_average, find_optimal_shift, generate_physical_labels, smooth_labels

def compute_vehicle_rotation(df_sm, df_vm):
    """
    Computes a 3x3 rotation matrix that maps the smartphone frame to the vehicle frame.
    """
    # Linear acceleration
    acc = np.vstack([
        df_sm['Ax'].values - df_sm['GravityX'].values,
        df_sm['Ay'].values - df_sm['GravityY'].values,
        df_sm['Az'].values - df_sm['GravityZ'].values
    ]).T # (N, 3)

    grav = np.vstack([
        df_sm['GravityX'].values,
        df_sm['GravityY'].values,
        df_sm['GravityZ'].values
    ]).T # (N, 3)

    obd_long_col = [c for c in df_vm.columns if 'Longitudinal Acceleration' in c][0]
    obd_long = df_vm[obd_long_col].values * 9.80665 # Convert g to m/s^2

    # 1. Z-axis is the average gravity vector (normalized)
    z_veh = np.mean(grav, axis=0)
    z_veh = z_veh / np.linalg.norm(z_veh)

    # 2. Project linear acceleration onto the plane perpendicular to Z
    dot_z = np.dot(acc, z_veh)
    acc_proj = acc - np.outer(dot_z, z_veh)

    # 3. Find X axis by correlating with OBD Longitudinal
    # Filter for robustness
    w = 50
    ax_f = moving_average(acc_proj[:, 0], w)
    ay_f = moving_average(acc_proj[:, 1], w)
    az_f = moving_average(acc_proj[:, 2], w)
    obd_f = moving_average(obd_long, w)

    # Ensure lengths match in case moving_average trims differently
    min_len = min(len(ax_f), len(obd_f))
    ax_f = ax_f[:min_len]
    ay_f = ay_f[:min_len]
    az_f = az_f[:min_len]
    obd_f = obd_f[:min_len]

    cov_x = np.mean(ax_f * obd_f) - np.mean(ax_f) * np.mean(obd_f)
    cov_y = np.mean(ay_f * obd_f) - np.mean(ay_f) * np.mean(obd_f)
    cov_z = np.mean(az_f * obd_f) - np.mean(az_f) * np.mean(obd_f)

    x_veh = np.array([cov_x, cov_y, cov_z])
    x_veh = x_veh / np.linalg.norm(x_veh)

    # 4. Y axis is cross product of Z and X
    y_veh = np.cross(z_veh, x_veh)
    y_veh = y_veh / np.linalg.norm(y_veh)

    # Rotation Matrix (Phone to Vehicle)
    R = np.vstack([x_veh, y_veh, z_veh]) # (3, 3)
    
    # Verify Rotation Matrix properties
    I_approx = np.dot(R, R.T)
    err = np.linalg.norm(I_approx - np.eye(3))
    det = np.linalg.det(R)
    if err > 1e-4 or abs(det - 1.0) > 1e-4:
        print(f"[WARNING] Invalid Rotation Matrix! ||R R^T - I|| = {err:.2e}, det(R) = {det:.5f}")
    
    return R

def create_vehicle_windows(df_sm, df_vm, shift, window_len=40, stride=10):
    if shift > 0:
        sm_aligned = df_sm.iloc[:-shift].reset_index(drop=True)
        vm_aligned = df_vm.iloc[shift:].reset_index(drop=True)
    elif shift < 0:
        sm_aligned = df_sm.iloc[-shift:].reset_index(drop=True)
        vm_aligned = df_vm.iloc[:shift].reset_index(drop=True)
    else:
        sm_aligned = df_sm.reset_index(drop=True)
        vm_aligned = df_vm.reset_index(drop=True)
        
    n_rows = min(len(sm_aligned), len(vm_aligned))
    if n_rows < window_len:
        return [], [], []
        
    sm_aligned = sm_aligned.iloc[:n_rows]
    vm_aligned = vm_aligned.iloc[:n_rows]
    
    # Clean S-M columns 
    sm_rename = {
        'ACCELEROMETER X': 'Ax',
        'ACCELEROMETER Y': 'Ay',
        'ACCELEROMETER Z': 'Az',
        'GRAVITY X': 'GravityX',
        'GRAVITY Y': 'GravityY',
        'GRAVITY Z': 'GravityZ',
        'GYROSCOPE Yaw': 'GyroYaw',
        'GYROSCOPE Pitch': 'GyroPitch',
        'GYROSCOPE Roll': 'GyroRoll'
    }
    for col in sm_aligned.columns:
        for k, v in sm_rename.items():
            if k in col:
                sm_aligned.rename(columns={col: v}, inplace=True)
                
    # 1. Compute Rotation Matrix
    R = compute_vehicle_rotation(sm_aligned, vm_aligned)
    
    # 2. Extract and Rotate Linear Acceleration
    acc = np.vstack([
        sm_aligned['Ax'].values - sm_aligned['GravityX'].values,
        sm_aligned['Ay'].values - sm_aligned['GravityY'].values,
        sm_aligned['Az'].values - sm_aligned['GravityZ'].values
    ]).T # (N, 3)
    acc_rotated = np.dot(acc, R.T) # (N, 3)
    
    # 3. Extract and Rotate Gyroscope
    gyro = np.vstack([
        sm_aligned['GyroYaw'].values, # Note: Check mapping if x/y/z yaw/pitch/roll mapping is different, but a 3D rotate handles this correctly assuming they align with Ax,Ay,Az axes
        sm_aligned['GyroPitch'].values,
        sm_aligned['GyroRoll'].values
    ]).T # (N, 3)
    gyro_rotated = np.dot(gyro, R.T) # (N, 3)
    
    # Extract Smartphone Gravity (Unrotated)
    grav = np.vstack([
        sm_aligned['GravityX'].values,
        sm_aligned['GravityY'].values,
        sm_aligned['GravityZ'].values
    ]).T # (N, 3)
    
    # 9 features: [Ax_v, Ay_v, Az_v, Gx_v, Gy_v, Gz_v, GravX_s, GravY_s, GravZ_s]
    features = np.hstack([acc_rotated, gyro_rotated, grav]) # (N, 9)
    
    raw_labels = generate_physical_labels(vm_aligned)
    events = smooth_labels(raw_labels, window_size=5)[:n_rows]
    
    speed_col = [c for c in vm_aligned.columns if 'Velocity' in c][0]
    speeds = vm_aligned[speed_col].values[:n_rows] / 3.6
    
    X_windows = []
    y_speed_windows = []
    y_event_windows = []
    
    for i in range(0, n_rows - window_len + 1, stride):
        end_idx = i + window_len - 1
        X_windows.append(features[i:i+window_len])
        y_speed_windows.append(speeds[end_idx])
        y_event_windows.append(events[end_idx])
        
    return X_windows, y_speed_windows, y_event_windows

def run_preprocessing():
    sm_dir = r"C:\Users\surwe\Project\INS\IO-VNBD\preprocess_data"
    vm_root = r"C:\Users\surwe\Project\INS\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset"
    out_dir = r"C:\Users\surwe\Project\INS\model\preprocess_data\vehicle_frame"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Scanning dataset drives...")
    s_files = [f for f in os.listdir(sm_dir) if f.startswith("S-") and f.endswith(".csv")]
    
    v_paths = {}
    for dirpath, dirnames, filenames in os.walk(vm_root):
        for f in filenames:
            if f.startswith("V-") and f.endswith(".csv"):
                name = f[2:-4]
                v_paths[name.lower()] = os.path.join(dirpath, f)
                
    pairs = []
    for sf in s_files:
        name = sf[2:-4]
        key = name.lower()
        if key in v_paths:
            pairs.append({
                'name': name,
                's_path': os.path.join(sm_dir, sf),
                'v_path': v_paths[key]
            })
        else:
            for k, p in v_paths.items():
                if k in key or key in k:
                    pairs.append({
                        'name': name,
                        's_path': os.path.join(sm_dir, sf),
                        'v_path': p
                    })
                    break
                    
    print(f"Matched {len(pairs)} drive sessions.")
    
    train_pairs = []
    val_pairs = []
    test_pairs = []
    
    for pair in pairs:
        name = pair['name']
        if name.startswith('S') and name != 'S-M':
            train_pairs.append(pair)
        elif name == 'M':
            train_pairs.append(pair)
        elif name.startswith('Y'):
            test_pairs.append(pair)
        elif name.startswith('Vfa'):
            test_pairs.append(pair)
        elif name.startswith('Vta'):
            num_str = ''.join([c for c in name if c.isdigit()])
            if num_str:
                num = int(num_str)
                if num <= 20:
                    train_pairs.append(pair)
                else:
                    val_pairs.append(pair)
            else:
                train_pairs.append(pair)
        elif name.startswith('Vtb'):
            num_str = ''.join([c for c in name if c.isdigit()])
            if num_str:
                num = int(num_str)
                if num <= 8:
                    train_pairs.append(pair)
                else:
                    val_pairs.append(pair)
            else:
                train_pairs.append(pair)
        elif name.startswith('Vw'):
            num_str = ''.join([c for c in name if c.isdigit()])
            if num_str:
                num = int(num_str)
                if num <= 10:
                    train_pairs.append(pair)
                elif num <= 14:
                    val_pairs.append(pair)
                else:
                    test_pairs.append(pair)
            else:
                train_pairs.append(pair)
        else:
            train_pairs.append(pair)
            
    print(f"Split counts - Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    
    def process_split(pairs_list, split_name):
        X_all, y_speed_all, y_event_all = [], [], []
        for i, pair in enumerate(pairs_list):
            print(f"[{split_name}] Processing {i+1}/{len(pairs_list)}: {pair['name']}...")
            try:
                df_sm = pd.read_csv(pair['s_path'])
                df_vm = pd.read_csv(pair['v_path'], encoding="latin-1")
                df_vm.columns = [c.strip() for c in df_vm.columns]
                
                # Fix S-M columns early for find_optimal_shift
                sm_rename = {'GYROSCOPE Pitch (rad/s)': 'GyroPitch'}
                for col in df_sm.columns:
                    for k, v in sm_rename.items():
                        if k in col:
                            df_sm.rename(columns={col: v}, inplace=True)
                            
                shift = find_optimal_shift(df_sm, df_vm)
                X_w, y_s_w, y_e_w = create_vehicle_windows(df_sm, df_vm, shift, window_len=40, stride=10)
                
                if len(X_w) > 0:
                    X_all.extend(X_w)
                    y_speed_all.extend(y_s_w)
                    y_event_all.extend(y_e_w)
            except Exception as e:
                print(f"Error processing {pair['name']}: {e}")
                
        return np.array(X_all, dtype=np.float32), np.array(y_speed_all, dtype=np.float32), np.array(y_event_all, dtype=np.int64)

    print("\nProcessing training set...")
    X_train, y_speed_train, y_event_train = process_split(train_pairs, "Train")
    
    print("\nProcessing validation set...")
    X_val, y_speed_val, y_event_val = process_split(val_pairs, "Val")
    
    print("\nProcessing test set...")
    X_test, y_speed_test, y_event_test = process_split(test_pairs, "Test")
    
    print(f"\nCreated windows - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    print("\nNormalizing datasets...")
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_speed_train.npy"), y_speed_train)
    np.save(os.path.join(out_dir, "y_event_train.npy"), y_event_train)
    
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(out_dir, "y_speed_val.npy"), y_speed_val)
    np.save(os.path.join(out_dir, "y_event_val.npy"), y_event_val)
    
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_speed_test.npy"), y_speed_test)
    np.save(os.path.join(out_dir, "y_event_test.npy"), y_event_test)
    
    np.save(os.path.join(out_dir, "mean.npy"), mean)
    np.save(os.path.join(out_dir, "std.npy"), std)
    
    print(f"\nAll preprocessed vehicle-frame datasets saved to: {out_dir}")

if __name__ == "__main__":
    run_preprocessing()
