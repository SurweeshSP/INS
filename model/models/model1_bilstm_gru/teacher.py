import torch
import torch.nn as nn
from models.common_heads import MultiTaskHead

class BiLSTMTeacher(nn.Module):
    """
    Teacher model using a deep Bi-LSTM to extract rich temporal features
    from the 20x6 smartphone IMU windows.
    Used only during development/training.
    """
    def __init__(self, input_dim=6, seq_len=20, hidden_dim_1=64, hidden_dim_2=32, num_classes=5):
        super(BiLSTMTeacher, self).__init__()
        
        # Bi-LSTM Layer 1: output size is hidden_dim_1 * 2
        self.bilstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim_1,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Bi-LSTM Layer 2: output size is hidden_dim_2 * 2
        self.bilstm2 = nn.LSTM(
            input_size=hidden_dim_1 * 2,
            hidden_size=hidden_dim_2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Linear layer mapping representations
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim_2 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Multi-Task Head
        self.heads = MultiTaskHead(input_dim=32, num_classes=num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) -> (batch, 20, 6)
        out1, _ = self.bilstm1(x)  # out1 shape: (batch, 20, hidden_dim_1 * 2)
        out2, _ = self.bilstm2(out1)  # out2 shape: (batch, 20, hidden_dim_2 * 2)
        
        # Take the hidden representation from the last sequence step
        last_step = out2[:, -1, :] # shape: (batch, hidden_dim_2 * 2)
        
        features = self.fc(last_step) # shape: (batch, 32)
        
        speed, log_var, motion_logits = self.heads(features)
        
        return speed, log_var, motion_logits, features
