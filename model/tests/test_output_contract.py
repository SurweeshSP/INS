import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from models.model_factory import get_model
from models.adaptive_controller import AdaptiveController

def test_output_contract():
    print("Running test_output_contract...")
    
    # 1. Load Model 2 and Model 4
    model2 = get_model('model2_student', input_dim=9)
    model4 = get_model('model4', input_dim=9)
    controller = AdaptiveController()
    
    model2.eval()
    model4.eval()
    
    # Input batch size 2, seq_len 40, channels 9
    x = torch.randn(2, 40, 9)
    
    with torch.no_grad():
        out2 = model2(x)
        out4 = model4(x)
        
        # Pass outputs to the Adaptive Controller
        fused = controller(out2, out4)
        
    # Check outputs contracts keys
    for out in [out2, out4, fused]:
        assert "speed_mps" in out
        assert "speed_variance" in out
        assert "confidence" in out
        assert "motion_logits" in out
        assert "motion_probs" in out
        assert "motion_class" in out
        
        assert out["speed_mps"].shape == (2,)
        assert out["speed_variance"].shape == (2,)
        assert out["confidence"].shape == (2,)
        assert out["motion_logits"].shape == (2, 5)
        assert out["motion_probs"].shape == (2, 5)
        assert out["motion_class"].shape == (2,)
        
        assert np.isfinite(out["speed_mps"][0].item())
        assert out["speed_variance"][0].item() >= 0.0
        assert 0.0 <= out["confidence"][0].item() <= 1.0
        assert out["motion_class"][0].item() in [0, 1, 2, 3, 4]
        
    print("test_output_contract passed successfully!")

if __name__ == '__main__':
    test_output_contract()
