import os
import torch
import numpy as np
from models.model_factory import get_model

# Load model checkpoint
script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(script_dir, "checkpoints", "best_accuracy_idr.pt")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Reconstruct correct model based on checkpoint metadata
    model_name = checkpoint.get("model_name", "best_accuracy_idr")
    if model_name == "ResNet1DAttentionModel":
        model = get_model('model4', **checkpoint["model_config"])
    else:
        model = get_model('best_accuracy_idr', **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Scale statistics
    mean = np.array(checkpoint["scaler"]["mean"], dtype=np.float32)
    std = np.array(checkpoint["scaler"]["std"], dtype=np.float32)
else:
    model = None
    mean = None
    std = None
    print(f"Warning: Checkpoint not found at {checkpoint_path}. Run training first.")

def predict(sm_window, timestamp=12.40):
    """
    Perform inference on a single 40x9 smartphone IMU window.
    sm_window: numpy array or torch tensor of shape (40, 9)
    """
    global model, mean, std
    if model is None:
        raise RuntimeError("Model checkpoint not loaded. Train the model first.")
        
    if not isinstance(sm_window, np.ndarray):
        sm_window = np.array(sm_window, dtype=np.float32)
        
    # Standardize input feature window using stored scaler
    sm_window_norm = (sm_window - mean) / std
    
    # Convert to Tensor and add batch dimension (1, 40, 9)
    x = torch.tensor(sm_window_norm, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        out = model(x)
        
    # Map index to class label
    class_to_label = {
        0: "NORMAL",
        1: "BRAKING",
        2: "ACCELERATING",
        3: "STOPPED",
        4: "TURNING"
    }
    
    motion_class_int = int(out["motion_class"][0].item())
    motion_class_str = class_to_label.get(motion_class_int, "NORMAL_CRUISING")
    
    return {
        "timestamp": timestamp,
        "speed_mps": round(float(out["speed_mps"][0].item()), 2),
        "speed_variance": round(float(out["speed_variance"][0].item()), 2),
        "confidence": round(float(out["confidence"][0].item()), 2),
        "motion_class": motion_class_str
    }
