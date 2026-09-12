import os
import torch
import torch.nn as nn

def apply_dynamic_quantization(model, device='cpu'):
    """
    Applies INT8 Dynamic Quantization to the model.
    Dynamically quantizes LSTM, GRU, and Linear layers, reducing weights to INT8.
    Ideal for recurrent networks like LSTM and GRU.
    """
    print(f"Applying INT8 dynamic quantization...")
    # Put model on CPU since PyTorch dynamic quantization is optimized for CPU
    model.to('cpu')
    model.eval()
    
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.LSTM, nn.GRU, nn.Linear},
        dtype=torch.qint8
    )
    print("INT8 dynamic quantization completed.")
    return quantized_model

def apply_fp16_quantization(model):
    """
    Converts model parameters to FP16 precision.
    Ideal for mobile GPU/DSP targets.
    """
    print("Applying FP16 precision conversion...")
    model.eval()
    # Convert all floating-point layers to half precision (FP16)
    fp16_model = model.half()
    print("FP16 conversion completed.")
    return fp16_model

def export_to_onnx(model, dummy_input, save_path, opset_version=15):
    """
    Exports a PyTorch model to ONNX format.
    """
    print(f"Exporting model to ONNX: {save_path}")
    model.eval()
    model.to('cpu')
    
    # Check shape of dummy input
    if isinstance(dummy_input, tuple):
         inp = tuple(d.to('cpu') for d in dummy_input)
    else:
         inp = dummy_input.to('cpu')
         
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        inp,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['speed', 'log_variance', 'motion_logits'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'speed': {0: 'batch_size'},
                      'log_variance': {0: 'batch_size'},
                      'motion_logits': {0: 'batch_size'}}
    )
    print(f"Model successfully saved to ONNX format.")
    
def get_model_size_kb(model_path):
    """
    Returns the file size of the saved model in KB.
    """
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        return round(size_bytes / 1024.0, 2)
    return 0.0
