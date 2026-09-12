import torch
import numpy as np

class AdaptiveFusionController:
    """
    Adaptive Fusion Controller for Two-Model Architecture.
    Favors Model B (Adaptive Dilated TCN) for low latency.
    Falls back to Model A (ResNet1D-Attention) when:
    - Model B uncertainty (speed variance) > threshold
    - Motion state transition is detected
    When fallback occurs, Model A is used for a cooldown period (e.g. 20 frames = 2 seconds).
    """
    def __init__(self, model_a, model_b, uncertainty_threshold=1.5, cooldown_frames=20):
        self.model_a = model_a
        self.model_b = model_b
        self.uncertainty_threshold = uncertainty_threshold
        self.cooldown_frames = cooldown_frames
        
        self.cooldown_counter = 0
        self.last_motion_state = None
        
    def predict(self, x):
        """
        x: S-M window tensor of shape (1, window_size, channels)
        """
        # Always run Model B (low latency, cheap)
        out_b = self.model_b(x)
        
        speed_b = out_b["speed_mps"].item()
        var_b = out_b["speed_variance"].item()
        motion_b = out_b["motion_class"].item()
        
        # Check conditions for Model A fallback
        high_uncertainty = var_b > self.uncertainty_threshold
        
        motion_transition = False
        if self.last_motion_state is not None and motion_b != self.last_motion_state:
            motion_transition = True
            
        self.last_motion_state = motion_b
        
        if high_uncertainty or motion_transition or self.cooldown_counter > 0:
            if high_uncertainty or motion_transition:
                # Reset cooldown if a new trigger occurs
                self.cooldown_counter = self.cooldown_frames
            else:
                self.cooldown_counter -= 1
                
            # Run Model A (High Accuracy)
            out_a = self.model_a(x)
            
            # Combine or select. Here we purely select Model A as requested by the architecture.
            return {
                "active_model": "A",
                "speed_mps": out_a["speed_mps"].item(),
                "speed_variance": out_a["speed_variance"].item(),
                "motion_class": out_a["motion_class"].item(),
                "confidence": out_a["confidence"].item()
            }
            
        else:
            # Model B is confident and stable
            return {
                "active_model": "B",
                "speed_mps": speed_b,
                "speed_variance": var_b,
                "motion_class": motion_b,
                "confidence": out_b["confidence"].item()
            }

if __name__ == "__main__":
    # Example usage mockup
    # model_a = ResNet1DAttentionModel(input_dim=6, seq_len=40)
    # model_b = CausalCNNStudent(input_dim=6)
    # fusion = AdaptiveFusionController(model_a, model_b)
    # dummy_input = torch.randn(1, 40, 6)
    # print(fusion.predict(dummy_input))
    pass
