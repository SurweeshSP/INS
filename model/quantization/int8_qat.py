import torch

def prepare_model_for_qat(model, backend='qnnpack'):
    """
    Prepares a floating point model for Quantization Aware Training (QAT).
    Inserts fake quantization modules.
    """
    model.eval()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    torch.ao.quantization.prepare_qat(model, inplace=True)
    return model

def convert_qat_model_to_int8(model):
    """
    Converts a trained QAT model into a quantized INT8 model.
    """
    model.eval()
    model.to('cpu')
    return torch.ao.quantization.convert(model, inplace=False)
