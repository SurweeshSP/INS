import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskHead(nn.Module):
    """
    Common Multi-Task Output Head for all student and teacher architectures.
    Produces:
    1. Speed prediction (mu) in m/s.
    2. Speed uncertainty (log_variance) s = log(sigma^2).
    3. Motion classification logits (Softmax applied in loss/inference).
    """
    def __init__(self, input_dim, num_classes=5):
        super(MultiTaskHead, self).__init__()
        # Shared representations
        self.fc_shared = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output heads
        self.speed_head = nn.Linear(input_dim, 1)
        self.variance_head = nn.Linear(input_dim, 1)
        self.motion_head = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # x is assumed to be the flattened hidden feature representation of shape (batch, input_dim)
        shared = self.fc_shared(x)
        
        speed = self.speed_head(shared)          # mu_t (m/s)
        log_var = self.variance_head(shared)      # s_t = log(sigma_t^2)
        motion_logits = self.motion_head(shared)  # motion class logits
        
        return speed, log_var, motion_logits

class HeteroscedasticLoss(nn.Module):
    """
    Computes heteroscedastic regression loss for speed estimation combined with uncertainty.
    Loss = 0.5 * (exp(-s) * (y_true - y_pred)^2 + s)
    where s = log(sigma^2).
    """
    def __init__(self, reduction='mean'):
        super(HeteroscedasticLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, speed_pred, log_var, speed_true):
        # Ensure correct shapes
        speed_pred = speed_pred.squeeze(-1)
        log_var = log_var.squeeze(-1)
        speed_true = speed_true.squeeze(-1)
        
        # exp(-s) = 1 / sigma^2
        precision = torch.exp(-log_var)
        squared_error = (speed_true - speed_pred) ** 2
        
        loss = 0.5 * (precision * squared_error + log_var)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class MultiTaskLoss(nn.Module):
    """
    Combines heteroscedastic regression loss for speed,
    cross-entropy loss for motion classification,
    and distillation loss if teacher targets are provided.
    """
    def __init__(self, lambda_v=1.0, lambda_u=0.5, lambda_m=0.5, lambda_d=1.0, class_weights=None):
        super(MultiTaskLoss, self).__init__()
        self.lambda_v = lambda_v
        self.lambda_u = lambda_u
        self.lambda_m = lambda_m
        self.lambda_d = lambda_d
        
        self.speed_loss_fn = nn.HuberLoss(delta=1.0) # Robust speed loss against GNSS outliers
        self.uncertainty_loss_fn = HeteroscedasticLoss()
        self.motion_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.distill_loss_fn = nn.MSELoss()
        
    def forward(self, outputs, targets, teacher_outputs=None, distill_mode=False):
        """
        outputs: tuple (speed_pred, log_var_pred, motion_logits)
        targets: tuple (speed_true, motion_true)
        teacher_outputs: tuple (teacher_speed_pred, teacher_features) or hidden features (h_T)
        distill_mode: bool, whether to compute distillation loss
        """
        speed_pred, log_var, motion_logits = outputs[:3]
        speed_true, motion_true = targets
        
        # 1. Primary Speed Loss (Huber)
        l_speed = self.speed_loss_fn(speed_pred.squeeze(-1), speed_true)
        
        # 2. Uncertainty Loss (Heteroscedastic)
        l_uncertainty = self.uncertainty_loss_fn(speed_pred, log_var, speed_true)
        
        # 3. Motion Classification Loss (Cross Entropy)
        l_motion = self.motion_loss_fn(motion_logits, motion_true)
        
        total_loss = self.lambda_v * l_speed + self.lambda_u * l_uncertainty + self.lambda_m * l_motion
        
        # 4. Optional Distillation Loss
        l_distill = 0.0
        if distill_mode and teacher_outputs is not None:
            if isinstance(teacher_outputs, tuple):
                # Target distillation on output (continuous speed)
                teacher_speed = teacher_outputs[0]
                l_distill = self.distill_loss_fn(speed_pred, teacher_speed)
            else:
                # Feature distillation on hidden state (h_T vs h_S)
                student_features = outputs[3] if len(outputs) > 3 else speed_pred # Fallback
                l_distill = self.distill_loss_fn(student_features, teacher_outputs)
            
            total_loss += self.lambda_d * l_distill
            
        return {
            "loss": total_loss,
            "speed_loss": l_speed.item(),
            "uncertainty_loss": l_uncertainty.item(),
            "motion_loss": l_motion.item(),
            "distill_loss": l_distill.item() if isinstance(l_distill, torch.Tensor) else l_distill
        }
