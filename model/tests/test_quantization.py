import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_factory import get_model

def test_quantization():
    print("Running test_quantization...")
    
    # 1. Quantize Model 2
    model2 = get_model('model2_student', input_dim=9)
    model2.eval()
    q_model2 = torch.ao.quantization.quantize_dynamic(
        model2,
        {nn.Linear, nn.Conv1d},
        dtype=torch.qint8
    )
    
    # Run inference on quantized model
    dummy_input = torch.randn(1, 40, 9)
    with torch.no_grad():
        out2 = q_model2(dummy_input)
    assert "speed_mps" in out2
    
    # 2. Quantize Model 4
    model4 = get_model('model4', input_dim=9)
    model4.eval()
    q_model4 = torch.ao.quantization.quantize_dynamic(
        model4,
        {nn.Linear, nn.Conv1d},
        dtype=torch.qint8
    )
    with torch.no_grad():
        out4 = q_model4(dummy_input)
    assert "speed_mps" in out4
    
    # 3. Size check on mock serialized state dict
    torch.save(q_model4.state_dict(), "temp_q.pth")
    size_kb = os.path.getsize("temp_q.pth") / 1024.0
    print(f"Quantized Model 4 state dict size: {size_kb:.2f} KB")
    os.remove("temp_q.pth")
    
    assert size_kb < 100.0, "Quantized model size exceeds 100 KB budget!"
    
    print("test_quantization passed successfully!")

if __name__ == '__main__':
    test_quantization()
