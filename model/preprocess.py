import os
import pandas as pd
import numpy as np

def moving_average(a, n=50):
    if len(a) <= n:
        n = max(1, len(a) // 4)
    if n <= 1:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def find_optimal_shift(df_sm, df_vm):
    # Find gyroscope and yaw rate columns
    gyro_pitch_col = [c for c in df_sm.columns if 'GyroPitch' in c][0]
    obd_yaw_col = [c for c in df_vm.columns if 'Yaw Rate' in c][0]
    
    gyro_pitch = df_sm[gyro_pitch_col].values * 180.0 / np.pi
    obd_yaw = df_vm[obd_yaw_col].values
    
    # Trim to common length to ensure slice lengths match
    n_common = min(len(gyro_pitch), len(obd_yaw))
    gyro_pitch = gyro_pitch[:n_common]
    obd_yaw = obd_yaw[:n_common]
    
    # Filter signals
    gyro_pitch_f = moving_average(gyro_pitch, 50)
    obd_yaw_f = moving_average(obd_yaw, 50)
    
    best_corr = -1
    best_shift = 0
    
    max_shift = min(100, len(gyro_pitch_f) // 3)
    for shift in range(-max_shift, max_shift):
        if shift < 0:
            a = gyro_pitch_f[-shift:]
            b = obd_yaw_f[:shift]
        elif shift > 0:
            a = gyro_pitch_f[:-shift]
            b = obd_yaw_f[shift:]
        else:
            a = gyro_pitch_f
            b = obd_yaw_f
            
        if len(a) == 0 or len(b) == 0:
            continue
        corr = abs(np.corrcoef(a, b)[0, 1])
        if np.isnan(corr):
            corr = 0
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
            
    return best_shift

def generate_physical_labels(df_vm):
    speed_col = [c for c in df_vm.columns if 'Velocity' in c][0]
    yaw_col = [c for c in df_vm.columns if 'Yaw Rate' in c][0]
    accel_col = [c for c in df_vm.columns if 'Longitudinal' in c][0]
    
    speeds = df_vm[speed_col].values
    yaws = df_vm[yaw_col].values
    accels = df_vm[accel_col].values
    
    # Define physically interpretable thresholds:
    # 1. STOPPED: speed <= 1.0 km/h (0.278 m/s)
    # 2. TURNING: abs(yaw) >= 3.0 deg/s
    # 3. ACCELERATING: accel >= 0.05 g
    # 4. BRAKING: accel <= -0.05 g
    # 5. NORMAL_CRUISING: otherwise
    labels = []
    for i in range(len(df_vm)):
        speed = speeds[i]
        yaw = abs(yaws[i])
        accel = accels[i]
        
        if speed <= 1.0:
            labels.append(3) # STOPPED
        elif yaw >= 3.0:
            labels.append(4) # TURNING
        elif accel >= 0.05:
            labels.append(2) # ACCELERATING
        elif accel <= -0.05:
            labels.append(1) # BRAKING
        else:
            labels.append(0) # NORMAL_CRUISING
    return np.array(labels, dtype=np.int64)

def smooth_labels(labels, window_size=5):
    # Apply rolling majority filter for temporal stabilization
    smoothed = np.copy(labels)
    half_w = window_size // 2
    for i in range(half_w, len(labels) - half_w):
        window = labels[i - half_w : i + half_w + 1]
        counts = np.bincount(window)
        majority = np.argmax(counts)
        smoothed[i] = majority
    return smoothed

def create_drive_windows(df_sm, df_vm, shift, window_len=40, stride=10):
    # Apply shift to align rows
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
        
    # Extract features (9 channels including gravity components)
    features = sm_aligned[['Ax', 'Ay', 'Az', 'GyroYaw', 'GyroPitch', 'GyroRoll', 'GravityX', 'GravityY', 'GravityZ']].values[:n_rows]
    
    # Generate labels physically from vehicle dynamics data
    raw_labels = generate_physical_labels(vm_aligned)
    events = smooth_labels(raw_labels, window_size=5)[:n_rows]
    
    # Target speeds (in m/s)
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
    out_dir = r"C:\Users\surwe\Project\INS\model\preprocess_data\windowed"
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
                    
    print(f"Matched {len(pairs)} drive sessions.")
    
    # 2. Partition pairs into splits
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
    
    # 3. Process and window each split
    def process_split(pairs_list, split_name):
        X_all, y_speed_all, y_event_all = [], [], []
        for i, pair in enumerate(pairs_list):
            print(f"[{split_name}] Processing {i+1}/{len(pairs_list)}: {pair['name']}...")
            try:
                df_sm = pd.read_csv(pair['s_path'])
                df_vm = pd.read_csv(pair['v_path'], encoding="latin-1")
                df_vm.columns = [c.strip() for c in df_vm.columns]
                
                # Find optimal shift
                shift = find_optimal_shift(df_sm, df_vm)
                
                # Generate windows (window_len = 40)
                X_w, y_s_w, y_e_w = create_drive_windows(df_sm, df_vm, shift, window_len=40, stride=10)
                
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
    
    # 4. Normalize windows using Train statistics
    print("\nNormalizing datasets...")
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    # 5. Save arrays
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
    
    print(f"\nAll preprocessed windowed datasets saved to: {out_dir}")

if __name__ == "__main__":
    run_preprocessing()
