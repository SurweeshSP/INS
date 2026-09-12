import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model_factory import get_model

def test_sm_inference():
    print("Running test_sm_inference...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.abspath(os.path.join(script_dir, "..", "checkpoints"))
    temp_ckpt_path = os.path.join(checkpoints_dir, "best_accuracy_idr.pt")
    
    # Generate mock checkpoint if not present
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
    
    has_temp = False
    if not os.path.exists(temp_ckpt_path):
        torch.save(mock_checkpoint, temp_ckpt_path)
        has_temp = True
        
    # Import predict and run inference
    import inference
    dummy_window = np.random.randn(40, 9)
    result = inference.predict(dummy_window)
    
    assert "timestamp" in result
    assert "speed_mps" in result
    assert "speed_variance" in result
    assert "confidence" in result
    assert "motion_class" in result
    
    assert isinstance(result["speed_mps"], float)
    assert isinstance(result["speed_variance"], float)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["motion_class"], str)
    
    # Clean up mock checkpoint
    if has_temp and os.path.exists(temp_ckpt_path):
        os.remove(temp_ckpt_path)
        
    print("test_sm_inference passed successfully!")

if __name__ == '__main__':
    test_sm_inference()
