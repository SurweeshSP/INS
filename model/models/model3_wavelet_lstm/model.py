import torch
import torch.nn as nn
from models.common_heads import MultiTaskHead

class WaveletLSTMModel(nn.Module):
    """
    Model 3: Wavelet-LSTM.
    Utilizes causal Haar Wavelet features as inputs (18 channels for 9-channel raw inputs).
    Features:
    1. LSTM (32 hidden state)
    2. Adaptive state gate (gating to 16 channels)
    3. Multi-Task Output Head
    """
    def __init__(self, input_dim=18, hidden_dim=32, num_classes=5):
        super(WaveletLSTMModel, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Adaptive state gate
        self.gate_fc = nn.Linear(hidden_dim, 16)
        self.proj_fc = nn.Linear(hidden_dim, 16)
        
        # Multi-Task Head
        self.heads = MultiTaskHead(input_dim=16, num_classes=num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) -> (batch, N, 18)
        out, _ = self.lstm(x)  # out shape: (batch, seq_len, hidden_dim)
        
        # Extract representation from the last time step
        last_step = out[:, -1, :]  # shape: (batch, hidden_dim)
        
        # Adaptive state gate
        gate = torch.sigmoid(self.gate_fc(last_step))
        features = self.proj_fc(last_step) * gate  # shape: (batch, 16)
        
        speed, log_var, motion_logits = self.heads(features)
        
        return speed, log_var, motion_logits, features
