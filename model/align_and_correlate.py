import pandas as pd
import numpy as np
from preprocess import find_optimal_shift

def moving_average(a, n=50):
    if len(a) <= n:
        n = max(1, len(a) // 4)
    if n <= 1:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

sm_raw_path = r"C:\Users\surwe\Project\INS\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset\M (Driver B)\S-M.csv"
vm_raw_path = r"C:\Users\surwe\Project\INS\IO-VNBD\Synchronised V abd S datasets\Categorised IOVNB Dataset\M (Driver B)\V-M.csv"

df_sm = pd.read_csv(sm_raw_path, encoding="latin-1")
df_vm = pd.read_csv(vm_raw_path, encoding="latin-1")

df_sm.columns = [c.strip() for c in df_sm.columns]
df_vm.columns = [c.strip() for c in df_vm.columns]

sm_rename = {
    'ACCELEROMETER X (m/s)': 'Ax',
    'ACCELEROMETER Y (m/s)': 'Ay',
    'ACCELEROMETER Z (m/s)': 'Az',
    'GRAVITY X (m/s)': 'GravityX',
    'GRAVITY Y (m/s)': 'GravityY',
    'GRAVITY Z (m/s)': 'GravityZ',
    'GYROSCOPE Yaw (rad/s)': 'GyroYaw',
    'GYROSCOPE Pitch (rad/s)': 'GyroPitch',
    'GYROSCOPE Roll (rad/s)': 'GyroRoll'
}
for col in df_sm.columns:
    for k, v in sm_rename.items():
        if k[:-1] in col:
            df_sm.rename(columns={col: v}, inplace=True)

shift = find_optimal_shift(df_sm, df_vm)
print(f"Optimal shift: {shift}")

if shift > 0:
    sm_aligned = df_sm.iloc[:-shift].reset_index(drop=True)
    vm_aligned = df_vm.iloc[shift:].reset_index(drop=True)
elif shift < 0:
    sm_aligned = df_sm.iloc[-shift:].reset_index(drop=True)
    vm_aligned = df_vm.iloc[:shift].reset_index(drop=True)
else:
    sm_aligned = df_sm.reset_index(drop=True)
    vm_aligned = df_vm.reset_index(drop=True)

n = min(len(sm_aligned), len(vm_aligned))
sm_aligned = sm_aligned.iloc[:n]
vm_aligned = vm_aligned.iloc[:n]

# Linear acceleration
acc = np.vstack([
    sm_aligned['Ax'].values - sm_aligned['GravityX'].values,
    sm_aligned['Ay'].values - sm_aligned['GravityY'].values,
    sm_aligned['Az'].values - sm_aligned['GravityZ'].values
]).T # (N, 3)

grav = np.vstack([
    sm_aligned['GravityX'].values,
    sm_aligned['GravityY'].values,
    sm_aligned['GravityZ'].values
]).T # (N, 3)

obd_long = vm_aligned[[c for c in df_vm.columns if 'Longitudinal Acceleration' in c][0]].values * 9.80665

# 1. Z-axis is the average gravity vector (normalized)
# Note: phone gravity sensor points opposite to gravity acceleration, so usually +9.8 on Z if flat
# We'll just define Z as the gravity direction.
z_veh = np.mean(grav, axis=0)
z_veh = z_veh / np.linalg.norm(z_veh)

# 2. Project linear acceleration onto the plane perpendicular to Z
# acc_proj = acc - (acc . z_veh) * z_veh
dot_z = np.dot(acc, z_veh)
acc_proj = acc - np.outer(dot_z, z_veh)

# 3. Find X axis by correlating with OBD Longitudinal
# Covariance between projected accel and OBD
cov_x = np.mean(acc_proj[:, 0] * obd_long) - np.mean(acc_proj[:, 0]) * np.mean(obd_long)
cov_y = np.mean(acc_proj[:, 1] * obd_long) - np.mean(acc_proj[:, 1]) * np.mean(obd_long)
cov_z = np.mean(acc_proj[:, 2] * obd_long) - np.mean(acc_proj[:, 2]) * np.mean(obd_long)

x_veh = np.array([cov_x, cov_y, cov_z])
x_veh = x_veh / np.linalg.norm(x_veh)

# 4. Y axis is cross product of Z and X
y_veh = np.cross(z_veh, x_veh)
y_veh = y_veh / np.linalg.norm(y_veh)

# Rotation Matrix (Phone to Vehicle)
R = np.vstack([x_veh, y_veh, z_veh]) # (3, 3)

# Rotate linear acceleration
acc_rotated = np.dot(acc, R.T) # (N, 3)

w = 50
ax_f = moving_average(acc_rotated[:, 0], w)
ay_f = moving_average(acc_rotated[:, 1], w)
az_f = moving_average(acc_rotated[:, 2], w)
obd_f = moving_average(obd_long, w)

print(f"Rotated Corr X: {np.corrcoef(ax_f, obd_f)[0, 1]:.4f}")
print(f"Rotated Corr Y: {np.corrcoef(ay_f, obd_f)[0, 1]:.4f}")
print(f"Rotated Corr Z: {np.corrcoef(az_f, obd_f)[0, 1]:.4f}")

print(f"Corr X: {np.corrcoef(ax_f, obd_f)[0, 1]:.4f}")
print(f"Corr Y: {np.corrcoef(ay_f, obd_f)[0, 1]:.4f}")
print(f"Corr Z: {np.corrcoef(az_f, obd_f)[0, 1]:.4f}")
