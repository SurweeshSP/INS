import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.model_factory import get_model

def test_best_accuracy_shapes():
    print("Running test_best_accuracy_shapes...")
    model = get_model('best_accuracy_idr', input_dim=9)
    model.eval()
    
    # Input batch size 2, seq_len 40, channels 9
    dummy_input = torch.randn(2, 40, 9)
    
    with torch.no_grad():
        out = model(dummy_input)
        
    assert out["speed_mps"].shape == (2,), f"Expected shape (2,), got {out['speed_mps'].shape}"
    assert out["speed_variance"].shape == (2,), f"Expected shape (2,), got {out['speed_variance'].shape}"
    assert out["confidence"].shape == (2,), f"Expected shape (2,), got {out['confidence'].shape}"
    assert out["motion_logits"].shape == (2, 5), f"Expected shape (2, 5), got {out['motion_logits'].shape}"
    assert out["motion_probs"].shape == (2, 5), f"Expected shape (2, 5), got {out['motion_probs'].shape}"
    assert out["motion_class"].shape == (2,), f"Expected shape (2,), got {out['motion_class'].shape}"
    
    print("test_best_accuracy_shapes passed successfully!")

if __name__ == '__main__':
    test_best_accuracy_shapes()
