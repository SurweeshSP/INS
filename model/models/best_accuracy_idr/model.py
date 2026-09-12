import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution wrapper to ensure no future leakage.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=0,
            **kwargs
        )
        
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        if self.padding > 0:
            x_padded = F.pad(x, (self.padding, 0)) # Pad only on the left side (time dimension)
        else:
            x_padded = x
        return self.conv(x_padded)

class ResidualTCNBlock(nn.Module):
    """
    Residual Temporal Convolutional Block.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(ResidualTCNBlock, self).__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.relu(out + self.shortcut(x))

class BestAccuracyIDR(nn.Module):
    """
    Best-Accuracy Smartphone INS Model.
    Consists of LayerNorm, Causal Stem, Residual TCN block, temporal attention,
    attention-weighted pooling, GLU-style adaptive gating, and multi-task heads.
    """
    def __init__(self, input_dim=9, num_classes=5, seq_len=40):
        super(BestAccuracyIDR, self).__init__()
        self.seq_len = seq_len
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Conv1D 9 -> 24 (kernel = 5)
        self.conv_stem = CausalConv1d(input_dim, 24, kernel_size=5)
        self.relu = nn.ReLU()
        
        # Residual TCN blocks
        self.tcn1 = ResidualTCNBlock(24, 24, kernel_size=3, dilation=1)
        self.tcn2 = ResidualTCNBlock(24, 24, kernel_size=3, dilation=2)
        self.tcn3 = ResidualTCNBlock(24, 32, kernel_size=3, dilation=4)
        
        # 2-head self attention
        self.self_attn = nn.MultiheadAttention(embed_dim=32, num_heads=2, batch_first=True)
        
        # Attention-weighted pooling projection
        self.pool_proj = nn.Linear(32, 1)
        
        # Adaptive Gates (GLU-style)
        self.speed_gate = nn.Linear(32, 32)
        self.speed_gate_proj = nn.Linear(32, 32)
        
        self.motion_gate = nn.Linear(32, 32)
        self.motion_gate_proj = nn.Linear(32, 32)
        
        self.uncertainty_gate = nn.Linear(32, 32)
        self.uncertainty_gate_proj = nn.Linear(32, 32)
        
        # Multi-task heads
        self.speed_head = nn.Linear(32, 1)
        self.motion_head = nn.Linear(32, num_classes)
        self.uncertainty_head = nn.Linear(32, 1)
        
    def forward(self, x):
        # Input shape: (batch, seq_len, input_dim) -> e.g. [B, 40, 9]
        # Layer norm over feature dimension
        x = self.layer_norm(x)
        
        # Permute to (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        
        # Stem and TCN blocks
        out = self.relu(self.conv_stem(x))
        out = self.tcn1(out)
        out = self.tcn2(out)
        out = self.tcn3(out) # shape: [B, 32, seq_len]
        
        # Attention prep: shape: [B, seq_len, 32]
        x_seq = out.permute(0, 2, 1)
        attn_out, _ = self.self_attn(x_seq, x_seq, x_seq)
        
        # Attention-weighted pooling
        scores = self.pool_proj(attn_out) # shape: [B, seq_len, 1]
        weights = F.softmax(scores, dim=1) # shape: [B, seq_len, 1]
        pooled = torch.sum(weights * attn_out, dim=1) # shape: [B, 32]
        
        # Adaptive Gating
        h_speed = self.speed_gate_proj(pooled) * torch.sigmoid(self.speed_gate(pooled))
        h_motion = self.motion_gate_proj(pooled) * torch.sigmoid(self.motion_gate(pooled))
        h_uncertainty = self.uncertainty_gate_proj(pooled) * torch.sigmoid(self.uncertainty_gate(pooled))
        
        # Heads
        speed_mps = self.speed_head(h_speed) # [B, 1]
        motion_logits = self.motion_head(h_motion) # [B, num_classes]
        raw_var = self.uncertainty_head(h_uncertainty) # [B, 1]
        
        # Post-processing outputs
        variance = F.softplus(raw_var) + 1e-4
        confidence = 1.0 / (1.0 + variance)
        
        motion_probs = F.softmax(motion_logits, dim=-1)
        motion_class = torch.argmax(motion_probs, dim=-1)
        
        return {
            "speed_mps": speed_mps.squeeze(-1),
            "speed_variance": variance.squeeze(-1),
            "confidence": confidence.squeeze(-1),
            "motion_logits": motion_logits,
            "motion_probs": motion_probs,
            "motion_class": motion_class
        }
