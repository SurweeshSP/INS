import torch
import torch.nn as nn
from models.common_heads import MultiTaskHead

class TransformerTeacher(nn.Module):
    """
    Teacher model using a Transformer Encoder (bi-directional)
    to extract powerful temporal representations from 20x6 windows.
    Used only during development/training.
    """
    def __init__(self, input_dim=6, seq_len=40, d_model=32, nhead=4, num_layers=2, num_classes=5):
        super(TransformerTeacher, self).__init__()
        
        # Projection layer: maps 6 IMU channels to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Dense pooling representation
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Multi-Task Head
        self.heads = MultiTaskHead(input_dim=d_model, num_classes=num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) -> (batch, 20, 6)
        x_proj = self.input_projection(x) + self.pos_encoder  # shape: (batch, 20, d_model)
        
        out = self.transformer_encoder(x_proj)  # shape: (batch, 20, d_model)
        
        # Average pooling across time sequence
        pooled = out.mean(dim=1)  # shape: (batch, d_model)
        
        features = self.fc(pooled) # shape: (batch, d_model)
        
        speed, log_var, motion_logits = self.heads(features)
        
        return speed, log_var, motion_logits, features
