import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.model2_transformer_cnn.teacher import TransformerTeacher
from models.model2_transformer_cnn.student import CausalCNNStudent
from models.common_heads import MultiTaskLoss

def train_transformer_teacher(train_loader, val_loader, epochs=10, lr=1e-3, device='cpu'):
    """
    Trains the Transformer Teacher model using standard supervised multi-task loss.
    """
    teacher = TransformerTeacher().to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = MultiTaskLoss()
    
    print("\n=== Training Transformer Teacher ===")
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

def train_cnn_student_with_distillation(teacher, train_loader, val_loader, epochs=15, lr=1e-3, lambda_d=1.0, device='cpu'):
    """
    Trains the Causal CNN Student model utilizing output speed distillation from the Transformer teacher.
    Loss includes speed loss, uncertainty loss, motion loss, and output distillation loss: L_distill = MSE(V_S, V_T)
    """
    teacher.to(device)
    teacher.eval() # Ensure teacher is in evaluation mode
    
    student = CausalCNNStudent().to(device)
    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = MultiTaskLoss(lambda_d=lambda_d)
    
    print("\n=== Training Causal CNN Student (Distillation) ===")
    for epoch in range(epochs):
        student.train()
        train_loss, val_loss = 0.0, 0.0
        
        for X_batch, y_batch, c_batch in train_loader:
            X_batch, y_batch, c_batch = X_batch.to(device), y_batch.to(device), c_batch.to(device)
            
            # Get teacher speed prediction
            with torch.no_grad():
                teacher_speed, _, _, _ = teacher(X_batch)
                
            optimizer.zero_grad()
            
            speed_pred, log_var, motion_logits, _ = student(X_batch)
            
            # Output-based distillation
            losses = loss_fn(
                outputs=(speed_pred, log_var, motion_logits),
                targets=(y_batch, c_batch),
                teacher_outputs=(teacher_speed,),
                distill_mode=True
            )
            loss = losses["loss"]
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        # Validation
        student.eval()
        with torch.no_grad():
            for X_batch, y_batch, c_batch in val_loader:
                X_batch, y_batch, c_batch = X_batch.to(device), y_batch.to(device), c_batch.to(device)
                
                speed_pred, log_var, motion_logits, _ = student(X_batch)
                teacher_speed, _, _, _ = teacher(X_batch)
                
                losses = loss_fn(
                    outputs=(speed_pred, log_var, motion_logits),
                    targets=(y_batch, c_batch),
                    teacher_outputs=(teacher_speed,),
                    distill_mode=True
                )
                val_loss += losses["loss"].item() * X_batch.size(0)
                
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    return student
