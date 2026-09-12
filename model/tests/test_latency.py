import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.model_factory import get_model
from training.train_two_models import profile_latency

def test_latency():
    print("Running test_latency...")
    
    # Verify latency of Model 4
    model = get_model('model4', input_dim=9)
    model.eval()
    
    # Profile over 10 warmups and 50 runs to keep test execution fast
    lat = profile_latency(model, input_shape=(1, 40, 9), warmups=10, runs=50)
    
    print(f"FP32 Mean Latency: {lat['mean']:.2f} ms")
    print(f"FP32 P99 Latency: {lat['p99']:.2f} ms")
    
    assert lat['mean'] < 20.0, "Model execution is too slow on CPU!"
    
    print("test_latency passed successfully!")

if __name__ == '__main__':
    test_latency()
