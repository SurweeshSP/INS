import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model3_wavelet_lstm.model import WaveletLSTMModel
from models.common_heads import MultiTaskLoss

def train_wavelet_lstm(train_loader, val_loader, epochs=15, lr=1e-3, qat_mode=False, device='cpu'):
    """
    Trains the Wavelet-LSTM model using standard multi-task loss.
    Optionally performs Quantization Aware Training (QAT) for the final epochs
    to simulate INT8 precision behavior.
    """
    model = WaveletLSTMModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = MultiTaskLoss()
    
    print(f"\n=== Training Wavelet-LSTM Model (QAT={qat_mode}) ===")
    
    if qat_mode:
        # Step 1: Set up quantization configurations for mobile deployment
        # PyTorch uses 'qnnpack' configuration for mobile CPU execution (XNNPACK/QNNPACK)
        try:
            model.eval()
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
            # Prepare model for Quantization Aware Training (inserts fake quantization modules)
            torch.ao.quantization.prepare_qat(model, inplace=True)
            model.to(device)
            print("Successfully prepared model for Quantization Aware Training (QAT).")
        except Exception as e:
            print(f"Warning: QAT preparation failed ({str(e)}). Falling back to float training.")
            qat_mode = False
            
    for epoch in range(epochs):
        model.train()
        train_loss, val_loss = 0.0, 0.0
        
        # In QAT mode, enable/disable observer and fake quantization if needed in later epochs
        if qat_mode and epoch > 3:
            # Freeze observers for calibration
            model.apply(torch.ao.quantization.disable_observer)
        if qat_mode and epoch > 5:
            # Freeze batch norm mean and variance estimates if any exist
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            
        for X_batch, y_batch, c_batch in train_loader:
            # For Wavelet LSTM, X_batch is already wavelet decomposed: shape (batch, 20, 12)
            X_batch, y_batch, c_batch = X_batch.to(device), y_batch.to(device), c_batch.to(device)
            
            optimizer.zero_grad()
            speed_pred, log_var, motion_logits, _ = model(X_batch)
            
            losses = loss_fn((speed_pred, log_var, motion_logits), (y_batch, c_batch))
            loss = losses["loss"]
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        # Validation
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch, c_batch in val_loader:
                X_batch, y_batch, c_batch = X_batch.to(device), y_batch.to(device), c_batch.to(device)
                speed_pred, log_var, motion_logits, _ = model(X_batch)
                losses = loss_fn((speed_pred, log_var, motion_logits), (y_batch, c_batch))
                val_loss += losses["loss"].item() * X_batch.size(0)
                
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    # If in QAT mode, convert the model to the quantized INT8 representation
    if qat_mode:
        try:
            model.eval()
            model.to('cpu') # Quantization conversion must happen on CPU
            quantized_model = torch.ao.quantization.convert(model, inplace=False)
            print("Successfully converted QAT model to quantized INT8 format.")
            return quantized_model
        except Exception as e:
            print(f"Error converting QAT model: {str(e)}")
            
    return model
