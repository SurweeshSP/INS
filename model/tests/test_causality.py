import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.model_factory import get_model

def test_causality():
    print("Running test_causality...")
    
    # 1. Test Model 2 (Adaptive Dilated TCN)
    model2 = get_model('model2_student', input_dim=9)
    model2.eval()
    
    dummy_input_a = torch.randn(1, 40, 9)
    dummy_input_b = dummy_input_a.clone()
    
    # Alter only the second half of the sequence (from index 20 onwards)
    dummy_input_b[:, 20:, :] = torch.randn(1, 20, 9)
    
    with torch.no_grad():
        out_a2 = model2(dummy_input_a)
        out_b2 = model2(dummy_input_b)
        
    # Standard output should be identical for indices < 20 because of causality
    # (Since global pooling aggregates the whole sequence, we check causality in TCN layers
    # by testing output at specific step index before global average pooling)
    # Let's test the TCN blocks directly to verify zero future leakage
    x_trans_a = dummy_input_a.permute(0, 2, 1)
    x_trans_b = dummy_input_b.permute(0, 2, 1)
    
    with torch.no_grad():
        tcn_out_a = model2.tcn_block2(model2.tcn_block1(model2.relu(model2.bn_stem(model2.depthwise_conv(model2.conv_1x1(x_trans_a))))))
        tcn_out_b = model2.tcn_block2(model2.tcn_block1(model2.relu(model2.bn_stem(model2.depthwise_conv(model2.conv_1x1(x_trans_b))))))
        
    diff2 = torch.abs(tcn_out_a[:, :, :20] - tcn_out_b[:, :, :20]).max().item()
    assert diff2 < 1e-5, f"Model 2 TCN causality violated! Max diff: {diff2}"
    
    # 2. Test Model 4 (ResNet1D + Attention)
    model4 = get_model('model4', input_dim=9)
    model4.eval()
    
    with torch.no_grad():
        stem_out_a = model4.res2(model4.res1(model4.relu(model4.bn_stem(model4.stem(x_trans_a)))))
        stem_out_b = model4.res2(model4.res1(model4.relu(model4.bn_stem(model4.stem(x_trans_b)))))
        
    diff4 = torch.abs(stem_out_a[:, :, :20] - stem_out_b[:, :, :20]).max().item()
    assert diff4 < 1e-5, f"Model 4 ResNet causality violated! Max diff: {diff4}"
    
    print("test_causality passed successfully!")

if __name__ == '__main__':
    test_causality()
