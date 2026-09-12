import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common_heads import MultiTaskHead
from models.model2_transformer_cnn.student import CausalConv1d

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet1DAttentionModel(nn.Module):
    """
    Model A: Compact ResNet1D + Attention model.
    Features:
    1. Causal Conv1D Stem (input -> 32)
    2. ResBlock 1 (32 -> 48)
    3. ResBlock 2 (48 -> 64)
    4. Lightweight Self-Attention (heads=2, embed_dim=64)
    5. Adaptive state gate
    6. Multi-Task Output Head
    """
    def __init__(self, input_dim=6, seq_len=40, d_model=32, num_heads=2, num_classes=5):
        super(ResNet1DAttentionModel, self).__init__()
        
        # Stem: Conv1d from input_dim to d_model (32)
        self.stem = CausalConv1d(input_dim, d_model, kernel_size=3)
        self.bn_stem = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        
        # ResBlock 1 (32 -> 48)
        downsample1 = nn.Sequential(
            CausalConv1d(d_model, 48, kernel_size=1),
            nn.BatchNorm1d(48)
        )
        self.res1 = ResidualBlock1D(d_model, 48, downsample=downsample1)
        
        # ResBlock 2 (48 -> 64)
        downsample2 = nn.Sequential(
            CausalConv1d(48, 64, kernel_size=1),
            nn.BatchNorm1d(64)
        )
        self.res2 = ResidualBlock1D(48, 64, downsample=downsample2)
        
        # Multi-Head Attention layer
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(64)
        
        # Dense mapping
        self.fc = nn.Linear(64, 32)
        
        # Adaptive state gate
        self.gate_fc = nn.Linear(32, 24)
        self.proj_fc = nn.Linear(32, 24)
        
        # Multi-Task Heads
        self.heads = MultiTaskHead(input_dim=24, num_classes=num_classes)
        
    def forward(self, x):
        # Input shape: (batch, seq_len, input_dim) -> (batch, N, 9)
        # Transpose to (batch, input_dim, seq_len)
        x_trans = x.permute(0, 2, 1)
        
        out = self.relu(self.bn_stem(self.stem(x_trans)))
        out = self.res1(out)
        out = self.res2(out) # shape: (batch, 64, seq_len)
        
        # Transpose back to (batch, seq_len, 64) for MultiheadAttention
        out_trans = out.permute(0, 2, 1)
        
        # Self-Attention
        attn_out, _ = self.attn(out_trans, out_trans, out_trans)
        out_trans = self.norm_attn(out_trans + attn_out) # residual connection + LayerNorm
        
        # Global Temporal Average Pooling
        pooled = out_trans.mean(dim=1) # shape: (batch, 64)
        
        # Dense representation
        fc_out = self.relu(self.fc(pooled))
        
        # Adaptive state gate
        gate = torch.sigmoid(self.gate_fc(fc_out))
        features = self.proj_fc(fc_out) * gate # shape: (batch, 24)
        
        speed, log_var, motion_logits = self.heads(features)
        
        variance = F.softplus(log_var) + 1e-4
        confidence = 1.0 / (1.0 + variance)
        motion_probs = F.softmax(motion_logits, dim=-1)
        motion_class = torch.argmax(motion_probs, dim=-1)
        
        return {
            "speed_mps": speed.squeeze(-1),
            "speed_variance": variance.squeeze(-1),
            "confidence": confidence.squeeze(-1),
            "motion_logits": motion_logits,
            "motion_probs": motion_probs,
            "motion_class": motion_class
        }
