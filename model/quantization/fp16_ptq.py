from models.quantization import apply_fp16_quantization

def post_training_quantization_fp16(model):
    """
    Wrapper for Post-Training FP16 Quantization.
    """
    return apply_fp16_quantization(model)
