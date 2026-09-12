import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model_factory import get_model

def test_checkpoint():
    print("Running test_checkpoint...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.abspath(os.path.join(script_dir, "..", "checkpoints"))
    os.makedirs(checkpoints_dir, exist_ok=True)
    temp_ckpt_path = os.path.join(checkpoints_dir, "best_accuracy_idr.pt")
    
    model = get_model('model4', input_dim=9)
    mock_checkpoint = {
        "model_name": "ResNet1DAttentionModel",
        "model_state_dict": model.state_dict(),
        "model_config": {"input_dim": 9, "seq_len": 40},
        "input_features": 9,
        "window_length": 40,
        "sampling_rate": 10,
        "motion_classes": ["NORMAL", "BRAKING", "ACCELERATING", "STOPPED", "TURNING"],
        "scaler": {"mean": np.zeros(9).tolist(), "std": np.ones(9).tolist()},
        "normalization": "z-score",
        "validation_metrics": {},
        "drift_metrics": {},
        "latency_metrics": {},
        "quantization_metrics": {},
        "training_config": {}
    }
    torch.save(mock_checkpoint, temp_ckpt_path)
    
    # Load and verify
    checkpoint = torch.load(temp_ckpt_path, map_location="cpu")
    
    required_keys = [
        "model_name",
        "model_state_dict",
        "model_config",
        "input_features",
        "window_length",
        "sampling_rate",
        "motion_classes",
        "scaler",
        "normalization",
        "validation_metrics",
        "drift_metrics",
        "latency_metrics",
        "quantization_metrics",
        "training_config"
    ]
    
    for key in required_keys:
        assert key in checkpoint, f"Missing required checkpoint key: {key}"
        
    print("test_checkpoint passed successfully!")

if __name__ == '__main__':
    test_checkpoint()
