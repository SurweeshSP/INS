import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tabulate import tabulate

from models.model_factory import get_model
from models.common_heads import MultiTaskLoss
from models.model3_wavelet_lstm.wavelet import apply_haar_wavelet_decomposition
from models.adaptive_controller import AdaptiveController

from quantization.int8_ptq import post_training_quantization_int8
from quantization.fp16_ptq import post_training_quantization_fp16

from models.evaluation.speed_metrics import compute_speed_metrics
from models.evaluation.motion_metrics import compute_motion_metrics
from models.evaluation.uncertainty import compute_uncertainty_metrics
from models.evaluation.latency import profile_latency
from models.evaluation.model_size import get_parameter_count

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def create_windows(df_sm, df_vm, window_len=40, stride=10):
    """
    Creates sliding windows from preprocessed S-M and V-M files.
    """
    features = df_sm[['Ax', 'Ay', 'Az', 'GyroYaw', 'GyroPitch', 'GyroRoll']].values
    timestamps = df_sm['timestamp'].values
    speeds = df_vm['Velocity_mps'].values
    events = df_vm['Event_Class'].values
    
    X, y_speed, y_event, window_times = [], [], [], []
    
    # Slide window
    for i in range(0, len(df_sm) - window_len + 1, stride):
        end_idx = i + window_len - 1
        X.append(features[i : i + window_len])
        # Target associated with the last sample of the window
        y_speed.append(speeds[end_idx])
        y_event.append(events[end_idx])
        window_times.append(timestamps[end_idx])
        
    return np.array(X, dtype=np.float32), np.array(y_speed, dtype=np.float32), np.array(y_event, dtype=np.int64), np.array(window_times, dtype=np.float32)

def train_epoch(model, loader, optimizer, loss_fn, device, distill_mode=False, teacher=None, model_type='standard'):
    model.train()
    total_loss = 0.0
    for X_batch, y_speed_batch, y_event_batch in loader:
        X_batch = X_batch.to(device)
        y_speed_batch = y_speed_batch.to(device)
        y_event_batch = y_event_batch.to(device)
        
        optimizer.zero_grad()
        
        if model_type == 'model1_student' and teacher is not None:
            # Slim GRU Student distillation
            with torch.no_grad():
                _, _, _, h_T = teacher(X_batch)
            speed_pred, log_var, motion_logits, h_S = model(X_batch)
            # We need to project h_S to match h_T dimension (16 -> 32)
            if not hasattr(model, 'proj_layer'):
                model.proj_layer = nn.Linear(16, 32).to(device)
                optimizer.add_param_group({'params': model.proj_layer.parameters()})
            h_S_projected = model.proj_layer(h_S)
            losses = loss_fn(
                outputs=(speed_pred, log_var, motion_logits, h_S_projected),
                targets=(y_speed_batch, y_event_batch),
                teacher_outputs=h_T,
                distill_mode=True
            )
        elif model_type == 'model2_student' and teacher is not None:
            # Causal CNN Student distillation
            with torch.no_grad():
                teacher_speed, _, _, _ = teacher(X_batch)
            speed_pred, log_var, motion_logits, _ = model(X_batch)
            losses = loss_fn(
                outputs=(speed_pred, log_var, motion_logits),
                targets=(y_speed_batch, y_event_batch),
                teacher_outputs=(teacher_speed,),
                distill_mode=True
            )
        else:
            # Standard training
            speed_pred, log_var, motion_logits, _ = model(X_batch)
            losses = loss_fn(
                outputs=(speed_pred, log_var, motion_logits),
                targets=(y_speed_batch, y_event_batch),
                distill_mode=False
            )
            
        loss = losses["loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        
    return total_loss / len(loader.dataset)

def evaluate_model(model, loader, device, model_name, wavelet_mode=False):
    model.eval()
    all_speed_preds, all_log_var_preds, all_event_preds = [], [], []
    all_speed_trues, all_event_trues = [], []
    
    # Check if model has float16 weights
    is_fp16 = False
    try:
        is_fp16 = next(model.parameters()).dtype == torch.float16
    except StopIteration:
        pass
        
    with torch.no_grad():
        for X_batch, y_speed_batch, y_event_batch in loader:
            X_batch = X_batch.to(device)
            if is_fp16:
                X_batch = X_batch.half()
            speed_pred, log_var, motion_logits, _ = model(X_batch)
            
            all_speed_preds.extend(speed_pred.cpu().float().squeeze(-1).numpy())
            all_log_var_preds.extend(log_var.cpu().float().squeeze(-1).numpy())
            all_event_preds.extend(torch.argmax(motion_logits, dim=-1).cpu().numpy())
            all_speed_trues.extend(y_speed_batch.numpy())
            all_event_trues.extend(y_event_batch.numpy())
            
    # Compute metrics
    speed_metrics = compute_speed_metrics(all_speed_trues, all_speed_preds)
    motion_metrics = compute_motion_metrics(all_event_trues, all_event_preds)
    uncertainty_metrics = compute_uncertainty_metrics(all_speed_trues, all_speed_preds, all_log_var_preds)
    
    return {
        "speed_preds": np.array(all_speed_preds),
        "log_var_preds": np.array(all_log_var_preds),
        "event_preds": np.array(all_event_preds),
        "speed_metrics": speed_metrics,
        "motion_metrics": motion_metrics,
        "uncertainty_metrics": uncertainty_metrics
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    windowed_dir = r"C:\Users\surwe\Project\INS\model\preprocess_data\windowed"
    
    if not os.path.exists(os.path.join(windowed_dir, "X_train.npy")):
        print("Preprocessed windowed files not found. Running preprocess.py first...")
        import preprocess
        preprocess.run_preprocessing()
        
    X_train = np.load(os.path.join(windowed_dir, "X_train.npy"))
    y_speed_train = np.load(os.path.join(windowed_dir, "y_speed_train.npy"))
    y_event_train = np.load(os.path.join(windowed_dir, "y_event_train.npy"))
    
    X_val = np.load(os.path.join(windowed_dir, "X_val.npy"))
    y_speed_val = np.load(os.path.join(windowed_dir, "y_speed_val.npy"))
    y_event_val = np.load(os.path.join(windowed_dir, "y_event_val.npy"))
    
    X_test = np.load(os.path.join(windowed_dir, "X_test.npy"))
    y_speed_test = np.load(os.path.join(windowed_dir, "y_speed_test.npy"))
    y_event_test = np.load(os.path.join(windowed_dir, "y_event_test.npy"))
    
    print(f"Loaded splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Convert to Tensors
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_speed_train), torch.tensor(y_event_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_speed_val), torch.tensor(y_event_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_speed_test), torch.tensor(y_event_test))
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Calculate class weights from training targets
    from collections import Counter
    counts = Counter(y_event_train)
    total = sum(counts.values())
    n_classes = 5
    class_weights = [total / (n_classes * counts.get(c, 1)) for c in range(n_classes)]
    class_weights = np.array(class_weights)
    # Normalize class weights so their mean is 1.0
    class_weights = class_weights / np.mean(class_weights)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Computed Class Weights:", class_weights)
    
    # MultiTaskLoss with class weights and higher classification scale (lambda_m = 5.0)
    loss_fn = MultiTaskLoss(lambda_v=1.0, lambda_u=0.5, lambda_m=5.0, class_weights=class_weights_tensor)
    
    # ==========================================
    # Train Model 1 (Slim GRU Student)
    # ==========================================
    print("\nTraining Model 1 Student (Slim GRU) directly...")
    m1_student = get_model('model1_student', input_dim=9).to(device)
    optimizer = optim.Adam(m1_student.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    for epoch in range(10):
        loss = train_epoch(m1_student, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        print(f"  Epoch {epoch+1}/10 - Loss: {loss:.4f}")
        
    # ==========================================
    # Train Model 2 (Adaptive Dilated TCN Student)
    # ==========================================
    print("\nTraining Model 2 Student (Adaptive Dilated TCN) directly...")
    m2_student = get_model('model2_student', input_dim=9).to(device)
    optimizer = optim.Adam(m2_student.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    for epoch in range(10):
        loss = train_epoch(m2_student, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        print(f"  Epoch {epoch+1}/10 - Loss: {loss:.4f}")
        
    # ==========================================
    # Train Model 3 (Wavelet -> LSTM)
    # ==========================================
    print("\nApplying Haar Wavelet Decomposition to windows...")
    X_train_wav = apply_haar_wavelet_decomposition(torch.tensor(X_train)).numpy()
    X_val_wav = apply_haar_wavelet_decomposition(torch.tensor(X_val)).numpy()
    X_test_wav = apply_haar_wavelet_decomposition(torch.tensor(X_test)).numpy()
    
    train_dataset_wav = TensorDataset(torch.tensor(X_train_wav), torch.tensor(y_speed_train), torch.tensor(y_event_train))
    val_dataset_wav = TensorDataset(torch.tensor(X_val_wav), torch.tensor(y_speed_val), torch.tensor(y_event_val))
    test_dataset_wav = TensorDataset(torch.tensor(X_test_wav), torch.tensor(y_speed_test), torch.tensor(y_event_test))
    
    train_loader_wav = DataLoader(train_dataset_wav, batch_size=128, shuffle=True)
    val_loader_wav = DataLoader(val_dataset_wav, batch_size=256, shuffle=False)
    test_loader_wav = DataLoader(test_dataset_wav, batch_size=256, shuffle=False)
    
    print("\nTraining Model 3 (Wavelet-LSTM)...")
    m3_model = get_model('model3', input_dim=18).to(device)
    optimizer = optim.Adam(m3_model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    for epoch in range(10):
        loss = train_epoch(m3_model, train_loader_wav, optimizer, loss_fn, device)
        scheduler.step()
        print(f"  Epoch {epoch+1}/10 - Loss: {loss:.4f}")
        
    # ==========================================
    # Train Model 4 (ResNet1D + Attention)
    # ==========================================
    print("\nTraining Model 4 (ResNet1D-Attention)...")
    m4_model = get_model('model4', input_dim=9).to(device)
    optimizer = optim.Adam(m4_model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    for epoch in range(10):
        loss = train_epoch(m4_model, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        print(f"  Epoch {epoch+1}/10 - Loss: {loss:.4f}")
        
    # ==========================================
    # Apply Quantization
    # ==========================================
    print("\nApplying Quantization...")
    # Student 1: Dynamic INT8 Quantization (Optimized for CPU)
    m1_student_q = post_training_quantization_int8(m1_student)
    
    # Student 2: Dynamic INT8 Quantization (Adaptive Dilated TCN)
    m2_student_q = post_training_quantization_int8(m2_student)
    
    # Model 3: Dynamic INT8 Quantization
    m3_model_q = post_training_quantization_int8(m3_model)
    
    # Model 4: Dynamic INT8 Quantization
    m4_model_q = post_training_quantization_int8(m4_model)
    
    # Save the models
    torch.save(m1_student_q.state_dict(), "m1_student_q.pth")
    torch.save(m2_student_q.state_dict(), "m2_student_q.pth")
    torch.save(m3_model_q.state_dict(), "m3_model_q.pth")
    torch.save(m4_model_q.state_dict(), "m4_model_q.pth")
    
    # ==========================================
    # Evaluate Models on Test Set
    # ==========================================
    print("\nRunning Evaluation...")
    m1_student_q.to('cpu')
    m2_student_q.to('cpu')
    m3_model_q.to('cpu')
    m4_model_q.to('cpu')
    
    cpu_test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    cpu_test_loader_wav = DataLoader(test_dataset_wav, batch_size=256, shuffle=False)
    
    eval_m1 = evaluate_model(m1_student_q, cpu_test_loader, 'cpu', 'Model 1 Student (Quantized)')
    eval_m2 = evaluate_model(m2_student_q, cpu_test_loader, 'cpu', 'Model 2 Student (Quantized)')
    eval_m3 = evaluate_model(m3_model_q, cpu_test_loader_wav, 'cpu', 'Model 3 LSTM (Quantized)', wavelet_mode=True)
    eval_m4 = evaluate_model(m4_model_q, cpu_test_loader, 'cpu', 'Model 4 ResNet-Attention (Quantized)')
    
    # ==========================================
    # Run Adaptive Controller
    # ==========================================
    print("\nRunning Adaptive Controller Fusion...")
    controller = AdaptiveController()
    
    # Collect predictions
    mu1 = torch.tensor(eval_m1["speed_preds"])
    log_var1 = torch.tensor(eval_m1["log_var_preds"])
    
    m1_logits = []
    with torch.no_grad():
        for X_batch, _, _ in cpu_test_loader:
            _, _, logits, _ = m1_student_q(X_batch)
            m1_logits.append(logits)
    m1_logits = torch.cat(m1_logits, dim=0)
    
    m2_logits = []
    with torch.no_grad():
        for X_batch, _, _ in cpu_test_loader:
            _, _, logits, _ = m2_student_q(X_batch)
            m2_logits.append(logits)
    m2_logits = torch.cat(m2_logits, dim=0)
    
    m3_logits = []
    with torch.no_grad():
        for X_batch, _, _ in cpu_test_loader_wav:
            _, _, logits, _ = m3_model_q(X_batch)
            m3_logits.append(logits)
    m3_logits = torch.cat(m3_logits, dim=0)
    
    m4_logits = []
    with torch.no_grad():
        for X_batch, _, _ in cpu_test_loader:
            _, _, logits, _ = m4_model_q(X_batch)
            m4_logits.append(logits)
    m4_logits = torch.cat(m4_logits, dim=0)
    
    fused = controller(
        (mu1, log_var1, m1_logits),
        (torch.tensor(eval_m2["speed_preds"]), torch.tensor(eval_m2["log_var_preds"]), m2_logits),
        (torch.tensor(eval_m3["speed_preds"]), torch.tensor(eval_m3["log_var_preds"]), m3_logits),
        (torch.tensor(eval_m4["speed_preds"]), torch.tensor(eval_m4["log_var_preds"]), m4_logits)
    )
    
    # Compute metrics for fused outputs
    fused_speed = fused["speed_mps"].numpy()
    fused_var = fused["speed_variance"].numpy()
    fused_log_var = fused["log_variance"].numpy()
    fused_classes = fused["motion_class"].numpy()
    
    fused_speed_metrics = compute_speed_metrics(y_speed_test, fused_speed)
    fused_motion_metrics = compute_motion_metrics(y_event_test, fused_classes)
    fused_uncertainty_metrics = compute_uncertainty_metrics(y_speed_test, fused_speed, fused_log_var)
    
    # ==========================================
    # Compute Distance Drift Metrics
    # ==========================================
    def compute_drift(true_speeds, pred_speeds, horizon_steps=300):
        drifts = []
        n = len(true_speeds)
        if n <= horizon_steps:
            return 0.0
        dt = 0.1 # 10 Hz sampling rate
        for i in range(0, n - horizon_steps, 50): # step by 50 to speed up calculation
            pred_dist = np.sum(pred_speeds[i : i + horizon_steps]) * dt
            true_dist = np.sum(true_speeds[i : i + horizon_steps]) * dt
            if true_dist > 1.0:
                drift = abs(pred_dist - true_dist) / true_dist
                drifts.append(drift)
        if len(drifts) == 0:
            return 0.0
        return np.mean(drifts) * 100.0 # Return as percentage
        
    drift30_m1 = compute_drift(y_speed_test, eval_m1["speed_preds"], horizon_steps=300)
    drift60_m1 = compute_drift(y_speed_test, eval_m1["speed_preds"], horizon_steps=600)
    
    drift30_m2 = compute_drift(y_speed_test, eval_m2["speed_preds"], horizon_steps=300)
    drift60_m2 = compute_drift(y_speed_test, eval_m2["speed_preds"], horizon_steps=600)
    
    drift30_m3 = compute_drift(y_speed_test, eval_m3["speed_preds"], horizon_steps=300)
    drift60_m3 = compute_drift(y_speed_test, eval_m3["speed_preds"], horizon_steps=600)
    
    drift30_m4 = compute_drift(y_speed_test, eval_m4["speed_preds"], horizon_steps=300)
    drift60_m4 = compute_drift(y_speed_test, eval_m4["speed_preds"], horizon_steps=600)
    
    drift30_fused = compute_drift(y_speed_test, fused_speed, horizon_steps=300)
    drift60_fused = compute_drift(y_speed_test, fused_speed, horizon_steps=600)
    
    # ==========================================
    # Profile Latencies and Size
    # ==========================================
    lat1 = profile_latency(m1_student_q, input_shape=(1, 40, 9), device='cpu')
    lat2 = profile_latency(m2_student_q, input_shape=(1, 40, 9), device='cpu')
    lat3 = profile_latency(m3_model_q, input_shape=(1, 40, 18), device='cpu')
    lat4 = profile_latency(m4_model_q, input_shape=(1, 40, 9), device='cpu')
    
    size1 = get_parameter_count(m1_student)
    size2 = get_parameter_count(m2_student)
    size3 = get_parameter_count(m3_model)
    size4 = get_parameter_count(m4_model)
    
    # Format parameters
    p1 = f"{size1['TotalParameters']:,}"
    p2 = f"{size2['TotalParameters']:,}"
    p3 = f"{size3['TotalParameters']:,}"
    p4 = f"{size4['TotalParameters']:,}"
    
    # ==========================================
    # Generate Final Summary Table
    # ==========================================
    headers = ["Metric", "Model 1 (GRU)", "Model 2 (TCN)", "Model 3 (Wavelet-LSTM)", "Model 4 (ResNet-Attn)", "Adaptive Fusion"]
    table_data = [
        ["Parameters", p1, p2, p3, p4, "-"],
        ["FP32 Model Size (KB)", "~40 KB", "~30 KB", "~60 KB", "~140 KB", "-"],
        ["Quantized size (KB)", f"{os.path.getsize('m1_student_q.pth')/1024:.2f} KB", f"{os.path.getsize('m2_student_q.pth')/1024:.2f} KB", f"{os.path.getsize('m3_model_q.pth')/1024:.2f} KB", f"{os.path.getsize('m4_model_q.pth')/1024:.2f} KB", "-"],
        ["Speed MAE (m/s)", eval_m1["speed_metrics"]["MAE"], eval_m2["speed_metrics"]["MAE"], eval_m3["speed_metrics"]["MAE"], eval_m4["speed_metrics"]["MAE"], fused_speed_metrics["MAE"]],
        ["Speed RMSE (m/s)", eval_m1["speed_metrics"]["RMSE"], eval_m2["speed_metrics"]["RMSE"], eval_m3["speed_metrics"]["RMSE"], eval_m4["speed_metrics"]["RMSE"], fused_speed_metrics["RMSE"]],
        ["R2 Score", eval_m1["speed_metrics"]["R2"], eval_m2["speed_metrics"]["R2"], eval_m3["speed_metrics"]["R2"], eval_m4["speed_metrics"]["R2"], fused_speed_metrics["R2"]],
        ["Motion Accuracy", eval_m1["motion_metrics"]["Accuracy"], eval_m2["motion_metrics"]["Accuracy"], eval_m3["motion_metrics"]["Accuracy"], eval_m4["motion_metrics"]["Accuracy"], fused_motion_metrics["Accuracy"]],
        ["Motion F1 Score", eval_m1["motion_metrics"]["F1-Score"], eval_m2["motion_metrics"]["F1-Score"], eval_m3["motion_metrics"]["F1-Score"], eval_m4["motion_metrics"]["F1-Score"], fused_motion_metrics["F1-Score"]],
        ["Uncertainty NLL", eval_m1["uncertainty_metrics"]["NLL"], eval_m2["uncertainty_metrics"]["NLL"], eval_m3["uncertainty_metrics"]["NLL"], eval_m4["uncertainty_metrics"]["NLL"], fused_uncertainty_metrics["NLL"]],
        ["PICP (95% CI)", eval_m1["uncertainty_metrics"]["PICP"], eval_m2["uncertainty_metrics"]["PICP"], eval_m3["uncertainty_metrics"]["PICP"], eval_m4["uncertainty_metrics"]["PICP"], fused_uncertainty_metrics["PICP"]],
        ["30 s Drift", f"{drift30_m1:.2f}%", f"{drift30_m2:.2f}%", f"{drift30_m3:.2f}%", f"{drift30_m4:.2f}%", f"{drift30_fused:.2f}%"],
        ["60 s Drift", f"{drift60_m1:.2f}%", f"{drift60_m2:.2f}%", f"{drift60_m3:.2f}%", f"{drift60_m4:.2f}%", f"{drift60_fused:.2f}%"],
        ["Mean Latency (ms)", lat1["Mean"], lat2["Mean"], lat3["Mean"], lat4["Mean"], "-"],
        ["P95 Latency (ms)", lat1["P95"], lat2["P95"], lat3["P95"], lat4["P95"], "-"],
        ["P99 Latency (ms)", lat1["P99"], lat2["P99"], lat3["P99"], lat4["P99"], "-"]
    ]
    
    print("\n" + "="*50)
    print("           FINAL MODEL COMPARISON SUMMARY")
    print("="*50)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*50)
    
    # Print sample deployable prediction JSON
    print("\nSample Deployable Prediction JSON:")
    import json
    class_to_label = {0: "NORMAL_CRUISING", 1: "BRAKING", 2: "ACCELERATING", 3: "STOPPED", 4: "TURNING"}
    sample_json = {
        "timestamp": 12.40,
        "speed_mps": round(float(fused_speed[0]), 2),
        "speed_variance": round(float(fused_var[0]), 2),
        "confidence": round(float(fused["confidence"][0].item()), 2),
        "motion_class": class_to_label[int(fused_classes[0])]
    }
    print(json.dumps(sample_json, indent=2))
    
if __name__ == "__main__":
    main()
