import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.adaptive_controller import AdaptiveController

def test_adaptive_controller():
    print("Running test_adaptive_controller...")
    controller = AdaptiveController()
    
    # Test case 1: High agreement, low variance (e.g. NORMAL_CRUISING)
    mu1 = torch.tensor([10.0])
    log_var1 = torch.tensor([-2.0]) # low variance
    logits1 = torch.tensor([[5.0, 1.0, 1.0, 1.0, 1.0]]) # high logit for class 0 (NORMAL)
    
    mu2 = torch.tensor([10.1])
    log_var2 = torch.tensor([-2.1])
    logits2 = torch.tensor([[4.8, 1.2, 0.9, 1.0, 1.1]])
    
    mu3 = torch.tensor([9.9])
    log_var3 = torch.tensor([-1.9])
    logits3 = torch.tensor([[5.2, 0.8, 1.1, 0.9, 1.0]])
    
    fused = controller((mu1, log_var1, logits1), (mu2, log_var2, logits2), (mu3, log_var3, logits3))
    
    print(f"High agreement speed: {fused['speed_mps'].item():.2f}, variance: {fused['speed_variance'].item():.4f}, confidence: {fused['confidence'].item():.4f}, motion class: {fused['motion_class'].item()}")
    
    assert abs(fused["speed_mps"].item() - 10.0) < 0.2
    assert fused["motion_class"].item() == 0 # NORMAL_CRUISING
    assert fused["confidence"].item() > 0.7 # High confidence
    
    # Test case 2: High disagreement (should increase variance / reduce confidence)
    mu1_d = torch.tensor([5.0])
    mu2_d = torch.tensor([15.0])
    mu3_d = torch.tensor([10.0])
    
    fused_d = controller((mu1_d, log_var1, logits1), (mu2_d, log_var2, logits2), (mu3_d, log_var3, logits3))
    
    print(f"Disagreement speed: {fused_d['speed_mps'].item():.2f}, variance: {fused_d['speed_variance'].item():.4f}, confidence: {fused_d['confidence'].item():.4f}")
    
    assert fused_d["speed_variance"].item() > fused["speed_variance"].item()
    assert fused_d["confidence"].item() < fused["confidence"].item()
    
    print("test_adaptive_controller passed successfully!")

if __name__ == '__main__':
    test_adaptive_controller()
