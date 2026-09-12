import torch
from models.model1_bilstm_gru.teacher import BiLSTMTeacher
from models.model1_bilstm_gru.student import SlimGRUStudent
from models.model2_transformer_cnn.teacher import TransformerTeacher
from models.model2_transformer_cnn.student import CausalCNNStudent
from models.model3_wavelet_lstm.model import WaveletLSTMModel
from models.model4_resnet_attention.model import ResNet1DAttentionModel

from models.best_accuracy_idr.model import BestAccuracyIDR

def get_model(model_name, **kwargs):
    """
    Model Factory to instantiate and return model objects.
    Supported model names:
    - 'model1_teacher', 'model1_student'
    - 'model2_teacher', 'model2_student'
    - 'model3' (Wavelet-LSTM)
    - 'model4' (ResNet1D-Attention)
    - 'best_accuracy_idr' (Compact TCN + Attention)
    """
    model_name = model_name.lower()
    if model_name == 'model1_teacher':
        return BiLSTMTeacher(**kwargs)
    elif model_name == 'model1_student':
        return SlimGRUStudent(**kwargs)
    elif model_name == 'model2_teacher':
        return TransformerTeacher(**kwargs)
    elif model_name == 'model2_student':
        return CausalCNNStudent(**kwargs)
    elif model_name == 'model3':
        return WaveletLSTMModel(**kwargs)
    elif model_name == 'model4':
        return ResNet1DAttentionModel(**kwargs)
    elif model_name == 'best_accuracy_idr' or model_name == 'best_accuracy':
        return BestAccuracyIDR(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
