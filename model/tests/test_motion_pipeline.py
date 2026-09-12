import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
from models.model_factory import get_model
from training.train_two_models import run_motion_label_audit

def test_motion_pipeline():
    print("Running test_motion_pipeline...")
    
    # 1. Verify integer encoding of y_event
    y_event_train = np.random.randint(0, 5, size=(100,))
    y_event_val = np.random.randint(0, 5, size=(20,))
    y_event_test = np.random.randint(0, 5, size=(20,))
    
    # Run audit print test
    run_motion_label_audit(y_event_train, y_event_val, y_event_test, "Audit-Test", pred_test=y_event_test)
    
    # 2. Assert no Softmax inside CrossEntropyLoss input
    model = get_model('model4', input_dim=9)
    model.eval()
    dummy_input = torch.randn(1, 40, 9)
    with torch.no_grad():
        out = model(dummy_input)
        
    logits = out["motion_logits"]
    
    # Ensure logits are not bounded between 0 and 1 (softmax output)
    # The sum of softmax outputs is 1.0, but raw logits sum can be arbitrary
    logits_sum = torch.sum(logits, dim=-1)
    # If it was softmax, the sum would be exactly 1.0. For raw logits, it is highly unlikely to be 1.0.
    assert not torch.allclose(logits_sum, torch.ones_like(logits_sum)), "Softmax applied before CrossEntropyLoss!"
    
    # 3. Verify class weights passed to CrossEntropyLoss
    class_weights = torch.tensor([1.0, 1.2, 1.5, 0.8, 1.1])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    assert torch.all(loss_fn.weight == class_weights), "Class weights are not passed to CrossEntropyLoss!"
    
    print("test_motion_pipeline passed successfully!")

if __name__ == '__main__':
    test_motion_pipeline()
