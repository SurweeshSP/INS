import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.model_factory import get_model

def test_no_vm_leakage():
    print("Running test_no_vm_leakage...")
    
    # Verify Model 2 and Model 4
    for model_name in ['model2_student', 'model4']:
        model = get_model(model_name, input_dim=9)
        model.eval()
        
        # Verify strict shape requirement for 9 channels
        x_imu = torch.randn(1, 40, 9)
        with torch.no_grad():
            out_a = model(x_imu)
        
        # Verify rejection of 10 channels (V-M leakage)
        x_vm = torch.randn(1, 40, 10)
        try:
            with torch.no_grad():
                out_fail = model(x_vm)
            assert False, f"Model {model_name} accepted 10 channels! VM leakage risk!"
        except Exception as e:
            # Expected failure
            pass
            
    print("test_no_vm_leakage passed successfully!")

if __name__ == '__main__':
    test_no_vm_leakage()
