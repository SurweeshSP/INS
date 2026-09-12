import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.model_factory import get_model

def test_model_shapes():
    print("Running test_model_shapes...")
    
    # Test shapes of Model 2 (Adaptive Dilated TCN Student)
    m2_student = get_model('model2_student', input_dim=9)
    dummy_input_9 = torch.randn(2, 40, 9)
    
    m2_student.eval()
    with torch.no_grad():
        out2 = m2_student(dummy_input_9)
        
    assert out2["speed_mps"].shape == (2,), f"Expected (2,), got {out2['speed_mps'].shape}"
    assert out2["speed_variance"].shape == (2,), f"Expected (2,), got {out2['speed_variance'].shape}"
    assert out2["confidence"].shape == (2,), f"Expected (2,), got {out2['confidence'].shape}"
    assert out2["motion_logits"].shape == (2, 5), f"Expected (2, 5), got {out2['motion_logits'].shape}"
    assert out2["motion_probs"].shape == (2, 5), f"Expected (2, 5), got {out2['motion_probs'].shape}"
    assert out2["motion_class"].shape == (2,), f"Expected (2,), got {out2['motion_class'].shape}"
    
    # Test shapes of Model 4 (Compact ResNet1D + Attention)
    m4_model = get_model('model4', input_dim=9)
    m4_model.eval()
    with torch.no_grad():
        out4 = m4_model(dummy_input_9)
        
    assert out4["speed_mps"].shape == (2,), f"Expected (2,), got {out4['speed_mps'].shape}"
    assert out4["speed_variance"].shape == (2,), f"Expected (2,), got {out4['speed_variance'].shape}"
    assert out4["confidence"].shape == (2,), f"Expected (2,), got {out4['confidence'].shape}"
    assert out4["motion_logits"].shape == (2, 5), f"Expected (2, 5), got {out4['motion_logits'].shape}"
    assert out4["motion_probs"].shape == (2, 5), f"Expected (2, 5), got {out4['motion_probs'].shape}"
    assert out4["motion_class"].shape == (2,), f"Expected (2,), got {out4['motion_class'].shape}"
    
    print("test_model_shapes passed successfully!")

if __name__ == '__main__':
    test_model_shapes()
