import torch
import torch.nn as nn
from models.common_heads import MultiTaskHead

class SlimGRUStudent(nn.Module):
    """
    Slim GRU student model optimized for real-time mobile CPU inference.
    Features:
    1. Linear stem projection (9 -> 24)
    2. LayerNorm
    3. GRU layer (24 -> 32)
    4. Adaptive state gate (sigmoidal GLU-style gating to 16)
    5. Multi-Task Output Head
    """
    def __init__(self, input_dim=9, hidden_dim=32, num_classes=5):
        super(SlimGRUStudent, self).__init__()
        
        # Stem projection and LayerNorm
        self.stem = nn.Linear(input_dim, 24)
        self.ln = nn.LayerNorm(24)
        self.relu = nn.ReLU()
        
        # GRU Layer
        self.gru = nn.GRU(
            input_size=24,
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
        # x shape: (batch, seq_len, input_dim) -> (batch, N, 9)
        out = self.relu(self.ln(self.stem(x)))
        out, _ = self.gru(out)  # out shape: (batch, seq_len, hidden_dim)
        
        # Take the hidden representation of the last time step
        last_step = out[:, -1, :]  # shape: (batch, hidden_dim)
        
        # Adaptive state gate
        gate = torch.sigmoid(self.gate_fc(last_step))
        features = self.proj_fc(last_step) * gate  # shape: (batch, 16)
        
        speed, log_var, motion_logits = self.heads(features)
        
        return speed, log_var, motion_logits, features
