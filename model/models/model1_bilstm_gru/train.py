import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.model1_bilstm_gru.teacher import BiLSTMTeacher
from models.model1_bilstm_gru.student import SlimGRUStudent
from models.common_heads import MultiTaskLoss

def train_teacher(train_loader, val_loader, epochs=10, lr=1e-3, device='cpu'):
    """
    Trains the Bi-LSTM Teacher model using standard supervised multi-task loss.
    """
    teacher = BiLSTMTeacher().to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = MultiTaskLoss()
    
    print("\n=== Training Bi-LSTM Teacher ===")
    for epoch in range(epochs):
        teacher.train()
        train_loss, val_loss = 0.0, 0.0
        
        for X_batch, y_batch, c_batch in train_loader:
            X_batch, y_batch, c_batch = X_batch.to(device), y_batch.to(device), c_batch.to(device)
            
            optimizer.zero_grad()
            speed_pred, log_var, motion_logits, _ = teacher(X_batch)
            
            losses = loss_fn((speed_pred, log_var, motion_logits), (y_batch, c_batch))
            loss = losses["loss"]
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        # Validation
        teacher.eval()
        with torch.no_grad():
            for X_batch, y_batch, c_batch in val_loader:
                X_batch, y_batch, c_batch = X_batch.to(device), y_batch.to(device), c_batch.to(device)
                speed_pred, log_var, motion_logits, _ = teacher(X_batch)
                losses = loss_fn((speed_pred, log_var, motion_logits), (y_batch, c_batch))
                val_loss += losses["loss"].item() * X_batch.size(0)
                
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    return teacher

def train_student_with_distillation(teacher, train_loader, val_loader, epochs=15, lr=1e-3, lambda_d=1.0, device='cpu'):
    """
    Trains the Slim GRU Student model utilizing feature-based distillation from the trained Bi-LSTM teacher.
    Loss includes speed loss, uncertainty loss, motion loss, and distillation loss: L_distill = MSE(h_T, h_S)
    """
    teacher.to(device)
    teacher.eval() # Ensure teacher is in evaluation mode
    
    student = SlimGRUStudent().to(device)
    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=1e-5)
    
    # We project student features (16-dim) to teacher features (32-dim) for feature distillation
    projection = nn.Linear(16, 32).to(device)
    optimizer_proj = optim.Adam(projection.parameters(), lr=lr)
    
    loss_fn = MultiTaskLoss(lambda_d=lambda_d)
    
    print("\n=== Training Slim GRU Student (Distillation) ===")
    for epoch in range(epochs):
        student.train()
        projection.train()
        train_loss, val_loss = 0.0, 0.0
        
        for X_batch, y_batch, c_batch in train_loader:
            X_batch, y_batch, c_batch = X_batch.to(device), y_batch.to(device), c_batch.to(device)
            
            # Get teacher hidden representations (32-dim)
            with torch.no_grad():
                _, _, _, h_T = teacher(X_batch)
                
            optimizer.zero_grad()
            optimizer_proj.zero_grad()
            
            speed_pred, log_var, motion_logits, h_S = student(X_batch)
            
            # Project student features to match teacher feature dimension
            h_S_projected = projection(h_S)
            
            losses = loss_fn(
                outputs=(speed_pred, log_var, motion_logits, h_S_projected),
                targets=(y_batch, c_batch),
                teacher_outputs=h_T,
                distill_mode=True
            )
            loss = losses["loss"]
            
            loss.backward()
            optimizer.step()
            optimizer_proj.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        # Validation
        student.eval()
        projection.eval()
        with torch.no_grad():
            for X_batch, y_batch, c_batch in val_loader:
                X_batch, y_batch, c_batch = X_batch.to(device), y_batch.to(device), c_batch.to(device)
                
                speed_pred, log_var, motion_logits, h_S = student(X_batch)
                _, _, _, h_T = teacher(X_batch)
                h_S_projected = projection(h_S)
                
                losses = loss_fn(
                    outputs=(speed_pred, log_var, motion_logits, h_S_projected),
                    targets=(y_batch, c_batch),
                    teacher_outputs=h_T,
                    distill_mode=True
                )
                val_loss += losses["loss"].item() * X_batch.size(0)
                
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    return student
