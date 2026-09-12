import torch
import torch.nn as nn
import numpy as np

class AdaptiveController(nn.Module):
    """
    Adaptive Two-Model Controller.
    Selectively fuses or switches between:
    - Model 2: Adaptive Dilated TCN (Fast Expert)
    - Model 4: Compact ResNet1D + Attention (Accuracy/Drift Expert)
    """
    def __init__(self, disagreement_threshold=1.5, confidence_threshold=0.7, epsilon=1e-5):
        super(AdaptiveController, self).__init__()
        self.disagreement_threshold = disagreement_threshold
        self.confidence_threshold = confidence_threshold
        self.epsilon = epsilon
        
    def forward(self, out_m2, out_m4, latency_critical=False):
        """
        out_m2: Dictionary output from Model 2 Causal TCN
        out_m4: Dictionary output from Model 4 ResNet-Attn
        latency_critical: Boolean flag. If True, favors Model 2 to optimize latency.
        """
        speed_m2 = out_m2["speed_mps"]
        var_m2 = out_m2["speed_variance"]
        conf_m2 = out_m2["confidence"]
        logits_m2 = out_m2["motion_logits"]
        
        speed_m4 = out_m4["speed_mps"]
        var_m4 = out_m4["speed_variance"]
        conf_m4 = out_m4["confidence"]
        logits_m4 = out_m4["motion_logits"]
        
        # Calculate speed disagreement |speed_M2 - speed_M4|
        disagreement = torch.abs(speed_m2 - speed_m4)
        
        # Compute inverse variance weights
        w_m2 = 1.0 / (var_m2 + self.epsilon)
        w_m4 = 1.0 / (var_m4 + self.epsilon)
        sum_w = w_m2 + w_m4
        
        # Default uncertainty-weighted speed and variance
        weighted_speed = (w_m2 * speed_m2 + w_m4 * speed_m4) / sum_w
        weighted_var = 1.0 / sum_w
        
        # 1. Initialize outputs
        fused_speed = torch.zeros_like(speed_m2)
        fused_var = torch.zeros_like(var_m2)
        fused_confidence = torch.zeros_like(conf_m2)
        fused_logits = torch.zeros_like(logits_m2)
        
        # 2. Apply Adaptive Policy element-wise (supporting batch mode)
        for i in range(speed_m2.shape[0]):
            d = disagreement[i].item()
            c2 = conf_m2[i].item()
            c4 = conf_m4[i].item()
            
            if latency_critical or (c2 > c4 and d >= self.disagreement_threshold):
                # Policy: favor Model 2 (Fast expert)
                fused_speed[i] = speed_m2[i]
                fused_var[i] = var_m2[i] + d**2
                fused_confidence[i] = conf_m2[i]
                fused_logits[i] = logits_m2[i]
            elif d >= self.disagreement_threshold and c4 >= c2:
                # Policy: favor Model 4 (Drift expert)
                fused_speed[i] = speed_m4[i]
                fused_var[i] = var_m4[i] + d**2
                fused_confidence[i] = conf_m4[i]
                fused_logits[i] = logits_m4[i]
            elif d < self.disagreement_threshold and c2 > self.confidence_threshold and c4 > self.confidence_threshold:
                # Policy: low disagreement, high confidence -> weighted fusion
                fused_speed[i] = weighted_speed[i]
                fused_var[i] = weighted_var[i] + d**2
                fused_confidence[i] = 1.0 / (1.0 + fused_var[i])
                fused_logits[i] = 0.5 * (logits_m2[i] + logits_m4[i])
            else:
                # Fallback: Default weighted average
                fused_speed[i] = weighted_speed[i]
                fused_var[i] = weighted_var[i] + d**2
                fused_confidence[i] = 1.0 / (1.0 + fused_var[i])
                fused_logits[i] = 0.5 * (logits_m2[i] + logits_m4[i])
                
        # Calculate motion class predictions
        motion_probs = torch.softmax(fused_logits, dim=-1)
        motion_class = torch.argmax(motion_probs, dim=-1)
        
        return {
            "speed_mps": fused_speed,
            "speed_variance": fused_var,
            "confidence": fused_confidence,
            "motion_logits": fused_logits,
            "motion_probs": motion_probs,
            "motion_class": motion_class
        }
