from models.quantization import apply_dynamic_quantization

def post_training_quantization_int8(model, device='cpu'):
    """
    Wrapper for Post-Training Dynamic INT8 Quantization.
    """
    return apply_dynamic_quantization(model, device=device)
