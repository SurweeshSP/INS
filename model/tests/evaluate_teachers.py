import os
import sys
sys.path.append(r"C:\Users\surwe\Project\INS\model")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.metrics import classification_report

from models.model_factory import get_model

def main():
    device = torch.device('cpu')
    windowed_dir = r"C:\Users\surwe\Project\INS\model\preprocess_data\windowed"
    
    X_val = np.load(os.path.join(windowed_dir, "X_val.npy"))
    y_speed_val = np.load(os.path.join(windowed_dir, "y_speed_val.npy"))
    y_event_val = np.load(os.path.join(windowed_dir, "y_event_val.npy"))
    
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_speed_val), torch.tensor(y_event_val))
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Load teachers
    # Wait, in pipeline.py did we save the teacher checkpoints?
    # Let's check if there are any saved teacher checkpoints.
    # In pipeline.py we only saved the student checkpoints!
    # "torch.save(m1_student_q.state_dict(), 'm1_student_q.pth')"
    # "torch.save(m3_model_q.state_dict(), 'm3_model_q.pth')"
    # But wait, did we save the teacher checkpoints?
    # Let's check pipeline.py to see if we saved the teacher models.
    # In line 271-272, we saw:
    # torch.save(m1_student_q.state_dict(), "m1_student_q.pth")
    # torch.save(m3_model_q.state_dict(), "m3_model_q.pth")
    # And m2_student_fp16 was saved to "m2_student_q.pth"
    # So the teacher checkpoints were NOT saved to disk! They were only trained in-memory.
    # That means we cannot evaluate them directly without training.
    
    # Let's check the size of the teachers in model_factory.py
    # Let's view models/model_factory.py to check their definitions
    with open(r"C:\Users\surwe\Project\INS\model\models\model_factory.py", 'r') as f:
        print(f.read())

if __name__ == '__main__':
    main()
