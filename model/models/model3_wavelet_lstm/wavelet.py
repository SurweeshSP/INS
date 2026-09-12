import torch
import numpy as np

def apply_haar_wavelet_decomposition(x):
    """
    Applies Haar Wavelet Decomposition to a batch of IMU windows.
    Input:
        x: PyTorch Tensor of shape (batch, 20, 6)
    Returns:
        PyTorch Tensor of shape (batch, 20, 12)
        For each of the 6 channels, computes:
          cA_t = (x_t + x_{t-1}) / sqrt(2)
          cD_t = (x_t - x_{t-1}) / sqrt(2)
        For t=0, x_{t-1} is assumed to be x_0 (replicated padding).
    """
    batch_size, seq_len, num_channels = x.shape
    
    # Pad on the left (time dimension) with replicated first element
    # x_padded shape: (batch, 21, 6)
    x_padded = torch.cat([x[:, :1, :], x], dim=1)
    
    # Compute approximation and detail components
    # x_t is x_padded[:, 1:] and x_{t-1} is x_padded[:, :-1]
    cA = (x_padded[:, 1:, :] + x_padded[:, :-1, :]) / np.sqrt(2.0)
    cD = (x_padded[:, 1:, :] - x_padded[:, :-1, :]) / np.sqrt(2.0)
    
    # Interleave approximation and detail components along the channel dimension
    # Resulting shape: (batch, 20, 12)
    # For each channel i, we have cA[:, :, i] and cD[:, :, i] side-by-side
    out = torch.zeros(batch_size, seq_len, num_channels * 2, device=x.device, dtype=x.dtype)
    for i in range(num_channels):
        out[:, :, 2*i] = cA[:, :, i]
        out[:, :, 2*i + 1] = cD[:, :, i]
        
    return out
