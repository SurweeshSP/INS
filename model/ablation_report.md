# Ablation Study and Final Acceptance Report - Two-Model Adaptive INS

This report documents the training configurations, validation metrics, ablation study findings, and edge performance profiling for the deployable **BestAccuracyIDR** (Model 4 ResNet-Attn) model and its Adaptive Fusion pipeline.

## 1. Ablation Study Results

The Model 4 network was evaluated across three different training objectives on the validation drive split:

| Training Objective | Speed MAE | Speed R² | Motion F1 | 30s Drift | 60s Drift |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Speed Loss** | 6.2568 m/s | 0.2128 | 0.1353 | 57.87% | 65.52% |
| **Speed + Temporal Loss** | 5.4420 m/s | 0.3778 | 0.1353 | 66.32% | 73.05% |
| **Speed + Temporal + Drift Loss (Full)** | 4.5571 m/s | 0.5099 | 0.2730 | 72.32% | 82.30% |

## 2. Final Model Acceptance Dashboard (Held-out Test Trajectories)

| Criterion | Target | Measured | Status |
| :--- | :--- | :--- | :--- |
| **Motion Accuracy** | >= 0.90 | 0.6187 | NOT ACHIEVED |
| **Motion Macro-F1** | >= 0.90 | 0.1529 | NOT ACHIEVED |
| **Speed R²** | >= 0.90 | -0.0871 | NOT ACHIEVED |
| **Speed MAE** | <= 1.0 m/s | 8.4194 m/s | NOT ACHIEVED |
| **10 s Drift** | <= 10% | 42.16% | NOT ACHIEVED |
| **30 s Drift** | <= 10% | 59.59% | NOT ACHIEVED |
| **60 s Drift** | <= 10% | 55.07% | NOT ACHIEVED |
| **120 s Drift** | <= 10% | 64.87% | NOT ACHIEVED |
| **P99 Latency** | < 10 ms | 88.12 ms | NOT ACHIEVED |
| **Quantized Size** | < 100 KB | 89.39 KB | ACHIEVED |
| **10 Hz Navigation** | >= 10 Hz | ACHIEVED | ACHIEVED |

## 3. Edge Latency and Parameter Footprints

* **Total Parameters**: 18,455
* **FP32 Checkpoint Size**: 95.20 KB
* **INT8 Quantized Size**: 89.39 KB

### Latency Profiles (on CPU):

| Model Format | Mean Latency | P50 Latency | P95 Latency | P99 Latency |
| :--- | :--- | :--- | :--- | :--- |
| **FP32** | 21.38 ms | 14.90 ms | 54.47 ms | 88.12 ms |
| **INT8 Quantized** | 26.02 ms | 18.08 ms | 63.46 ms | 116.94 ms |

## 4. Comparison with Baseline Models

| Metric | Model 1 (Slim GRU) | Model 2 (Adaptive TCN) | Model 3 (Wavelet-LSTM) | Model 4 (ResNet-Attn) | Adaptive Fusion | **BestAccuracyIDR** (Ours) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Parameters** | 7,303 | 8,167 | 8,103 | 37,063 | - | **18,455** |
| **Quantized Size** | 20.19 KB | 45.30 KB | 19.11 KB | 151.86 KB | - | **89.39 KB** |
| **Speed MAE (m/s)** | 4.97 m/s | 4.80 m/s | 5.11 m/s | 4.31 m/s | 4.62 m/s | **8.42 m/s** *(target <= 1.0)* |
| **R² Score** | 0.58 | 0.61 | 0.56 | 0.67 | 0.63 | **-0.09** *(target >= 0.90)* |
| **Motion Accuracy** | 0.57 | 0.56 | 0.55 | 0.53 | 0.57 | **0.6187** *(target >= 0.90)* |
| **Motion F1 Score** | 0.58 | 0.57 | 0.57 | 0.57 | 0.58 | **0.1529** *(target >= 0.90)* |
| **30 s integrated Drift** | 21.97% | 20.35% | 22.62% | 16.58% | 19.22% | **59.59%** *(target <= 10%)* |
| **60 s integrated Drift** | 16.27% | 15.36% | 17.61% | 11.68% | 13.95% | **55.07%** *(target <= 10%)* |
| **Mean Latency (ms)** | 11.80 ms | 4.82 ms | 10.78 ms | 5.69 ms | - | **26.02 ms** |
| **P99 Latency (ms)** | 22.73 ms | 8.87 ms | 21.47 ms | 14.63 ms | - | **116.94 ms** |
