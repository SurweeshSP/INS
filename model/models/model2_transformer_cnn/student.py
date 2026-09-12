import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common_heads import MultiTaskHead

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
            x_padded = F.pad(x, (self.padding, 0)) # Pad only on the left side
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
        
        res = self.shortcut(x)
        return self.relu(out + res)

class CausalCNNStudent(nn.Module):
    """
    Adaptive Dilated TCN Student (replacing Causal CNN).
    Optimized for high-speed parallel causal sequence processing.
    """
    def __init__(self, input_dim=6, num_classes=5, seq_len=40, **kwargs):
        super(CausalCNNStudent, self).__init__()
        
        # 1x1 projection stem
        self.conv_1x1 = nn.Conv1d(input_dim, 16, kernel_size=1)
        
        # Depthwise causal Conv1D
        self.depthwise_conv = CausalConv1d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            dilation=1,
            groups=16
        )
        self.bn_stem = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        
        # Dilated residual TCN blocks
        self.tcn_block0 = ResidualTCNBlock(16, 16, kernel_size=3, dilation=1)
        self.tcn_block1 = ResidualTCNBlock(16, 32, kernel_size=3, dilation=2)
        self.tcn_block2 = ResidualTCNBlock(32, 32, kernel_size=3, dilation=4)
        
        # Dense mapping
        self.fc = nn.Linear(32, 32)
        
        # Adaptive state gate
        self.gate_fc = nn.Linear(32, 16)
        self.proj_fc = nn.Linear(32, 16)
        
        # Multi-task head
        self.heads = MultiTaskHead(input_dim=16, num_classes=num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) -> (batch, N, 9)
        # Transpose to (batch, input_dim, seq_len)
        x_trans = x.permute(0, 2, 1)
        
        # Apply stem
        out = self.conv_1x1(x_trans)
        out = self.relu(self.bn_stem(self.depthwise_conv(out)))
        
        # Apply residual TCN blocks
        out = self.tcn_block0(out)
        out = self.tcn_block1(out)
        out = self.tcn_block2(out) # shape: (batch, 32, seq_len)
        
        # Global Temporal Average Pooling
        pooled = out.mean(dim=-1) # shape: (batch, 32)
        
        # Dense projection
        fc_out = self.relu(self.fc(pooled))
        
        # Adaptive state gate
        gate = torch.sigmoid(self.gate_fc(fc_out))
        features = self.proj_fc(fc_out) * gate # shape: (batch, 16)
        
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
