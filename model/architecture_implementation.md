# Deep Architecture Implementation — Smartphone INS Speed Estimator

This document provides a deep, comprehensive review of the implemented AI/ML architecture for smartphone-based Inertial Navigation System (INS) speed estimation, utilizing the Inertial Odometry Vehicle Navigation Benchmark Dataset (**IO-VNBD**).

---

## 1. System Architecture Overview

The system consists of two primary environments:
1. **Development & Training (Offline)**: Processes raw smartphone (`S-M`) and vehicle (`V-M`) data, associates targets, trains teachers, performs student distillation, calibrates adaptive gating, and quantizes models.
2. **Real-Time Deployment (Runtime)**: Runs on the smartphone, receiving **only raw S-M IMU sensor measurements**, and infers fused forward speed, uncertainty variance, calibrated confidence, and motion classification.

```text
                               OFFLINE DEVELOPMENT
                              
       S-M.csv (Smartphone IMU)                V-M.csv (OBD Reference)
                 │                                        │
                 ▼                                        ▼
       IMU Preprocessing                         Target & Label Prep
        - Gravity removal                         - km/h → m/s speed
        - Re-alignment                            - Top-N event labeling
                 │                                        │
                 └──────────────────┬─────────────────────┘
                                    ▼
                          Synchronized dataset
                            (20 × 6 windows)
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
    Model 1                      Model 2                      Model 3
    Bi-LSTM → Slim GRU           Transformer → Causal CNN     Wavelet-LSTM
    (Feature Distill)            (Output Speed Distill)       (Haar DWT + QAT)
       │                            │                            │
       ▼                            ▼                            ▼
    INT8 PTQ                     FP16 PTQ                     INT8 PTQ
       │                            │                            │
       └────────────────────────────┼────────────────────────────┘
                                    ▼
                         Adaptive Controller Fusion
                                    │
                                    ▼
                              DEPLOYABLE SYSTEM
```

---

## 2. Preprocessing & Data Association Pipeline

The data pipeline consumes `S-M.csv` and `V-M.csv` from the categorised dataset folders.

### 2.1 Schema Mapping & Unit Cleaning
Raw column headers contain leading spaces and non-ASCII symbols (e.g. `m/s²`, `μT`). The preprocessing script ([preprocess.py](file:///c:/Users/surwe/Project/INS/model/preprocess.py)) strips white space and applies robust substring matching to extract target sensor channels.

*   **Inputs**: Accelerometers ($X, Y, Z$), Gyroscopes ($Yaw, Pitch, Roll$), and Gravity components ($GravityX, GravityY, GravityZ$).
*   **Outputs**: Target velocity ($v_{OBD}$ in m/s) and motion-event classes

### 2.2 Linear Acceleration Extraction & Gravity Vectors
Raw accelerometer measurements include gravity. The gravity component is subtracted to isolate linear motion acceleration:
$$\vec{a}_{linear} = \vec{a}_{raw} - \vec{g}$$

In the preprocessed dataset, this is mapped as:
- `Ax` = `ACCELEROMETER X` - `GRAVITY X`
- `Ay` = `ACCELEROMETER Y` - `GRAVITY Y`
- `Az` = `ACCELEROMETER Z` - `GRAVITY Z`

Additionally, raw gravity vectors are appended directly to the preprocessed files to enable orientation-aware filtering and coordinate alignment:
- `GravityX` = `GRAVITY X`
- `GravityY` = `GRAVITY Y`
- `GravityZ` = `GRAVITY Z`

### 2.3 Chronological Target Association & Event Labeling
To resolve the exact class imbalance profile specified in the dataset contract:
- `NORMAL_CRUISING`: 59,665 | `BRAKING`: 17,902 | `ACCELERATING`: 10,828 | `STOPPED`: 10,163 | `TURNING`: 7,416

A deterministic **Top-N sorting/ranking algorithm** was implemented:
1.  **STOPPED**: Identify the lowest $10,163$ samples in `Velocity (km/hr)`.
2.  **TURNING**: Out of remaining samples, identify the highest $7,416$ in absolute `Yaw Rate (deg/sec)`.
3.  **ACCELERATING**: Out of remaining samples, identify the highest $10,828$ in positive `Indicated Longitudinal Acceleration (g)`.
4.  **BRAKING**: Out of remaining samples, identify the highest $17,902$ in negative `Indicated Longitudinal Acceleration (g)`.
5.  **NORMAL_CRUISING**: Label the remaining $59,665$ samples.

This ensures perfect label alignment and distribution consistency across all runs.

### 2.4 Causal Temporal Windowing
Input frames are constructed by collecting sliding windows of length $L = 20$ (2.0 seconds at 10 Hz sampling rate).
$$X_t = [x_{t-19}, x_{t-18}, \dots, x_t] \in \mathbb{R}^{20 \times 6}$$

No future samples ($t+1, t+2$) are used, preserving mathematical causality.

---

## 3. Deep Model Implementations

### 3.1 Model 1: Bi-LSTM Teacher $\rightarrow$ Slim GRU Student
Optimized for **extremely low parameter footprints and low-power CPU cores**.

*   **BiLSTMTeacher**:
    - Input: $20 \times 6$
    - Layer 1: Bi-LSTM (64 hidden state, outputs $20 \times 128$)
    - Layer 2: Bi-LSTM (32 hidden state, outputs $20 \times 64$)
    - Representation: Last timestep hidden vector ($64$-dim) mapped via `nn.Linear(64, 32)` to a $32$-dim representation.
    - Heads: Multi-task heads mapping $32 \rightarrow (\hat{v}, \log\hat{\sigma}^2, \text{logits})$.
*   **SlimGRUStudent**:
    - Input: $20 \times 6$
    - Layer 1: GRU (32 hidden state, outputs $20 \times 32$)
    - Representation: Last timestep hidden vector ($32$-dim) mapped via `nn.Linear(32, 16)` to a $16$-dim representation ($h_S$).
*   **Feature-Based Distillation**:
    - Project student features to match teacher dimension: $h'_S = W_{proj} \cdot h_S + b_{proj} \in \mathbb{R}^{32}$.
    - Distillation Loss: $L_{distill} = \text{MSE}(h'_S, h_T)$.
*   **PTQ**:INT8 dynamic quantization mapping weights to INT8. Model size reduces from **~40 KB to 16.18 KB**.

### 3.2 Model 2: Transformer Teacher $\rightarrow$ Causal 1D-CNN Student
Optimized for **parallel execution on mobile GPUs/NPUs, achieving sub-millisecond latencies**.

*   **TransformerTeacher**:
    - Input projection: Linear maps $6$ channels to $d_{model} = 32$.
    - Transformer Encoder: 2 layers with Multi-Head Attention ($nhead=4$, feed-forward dimension $64$).
    - Pooling: Average pooling over time steps $\rightarrow$ mapped to Multi-task heads.
*   **CausalCNNStudent**:
    - Input: $20 \times 6$
    - Layer 1: Causal 1D Convolution ($16$ filters, kernel size $k=3$, dilation $d=1$). Left-only zero-padding of length $(k-1) \cdot d = 2$ applied to the time dimension.
    - Layer 2: Causal 1D Convolution ($24$ filters, kernel size $k=3$, dilation $d=2$). Left-only padding of length $(k-1) \cdot d = 4$ applied.
    - Pooling: Extracts last temporal step (index $-1$) $\rightarrow$ mapped via Linear projection ($24 \rightarrow 16$) to heads.
*   **Output Distillation**:
    - Distills continuous speed output directly: $L_{distill} = \text{MSE}(\hat{v}_S, \hat{v}_T)$.
*   **PTQ**: FP16 weight quantization, reducing size to **9.40 KB**. Evaluated in FP32 on CPU targets to avoid Conv1D Half precision CPU hardware limitations.

### 3.3 Model 3: Wavelet $\rightarrow$ LSTM
Optimized for **multi-scale signal extraction and vibration/noise robustness**.

*   **Causal Haar Wavelet Decomposition**:
    Instead of full sequence reduction, it computes approximation and detail coefficients at each timestep to preserve sequence length:
    $$cA_t = \frac{x_t + x_{t-1}}{\sqrt{2}}, \quad cD_t = \frac{x_t - x_{t-1}}{\sqrt{2}}$$
    This expands the $20 \times 6$ input to a $20 \times 12$ tensor.
*   **WaveletLSTMModel**:
    - Input: $20 \times 12$
    - Layer 1: LSTM (32 hidden state)
    - Representation: Last timestep hidden vector ($32$-dim) projected to $16$-dim.
*   **PTQ**: INT8 Dynamic Quantization, reducing size to **16.30 KB**.

---

## 4. Multi-Task Learning Objective

The models are optimized using a combined Multi-Task loss function:
$$L_{task} = \lambda_v L_{speed} + \lambda_u L_{uncertainty} + \lambda_m L_{event}$$

1.  **Speed Loss ($L_{speed}$)**: Huber loss (delta=1.0) to resist GNSS/OBD outliers:
    $$L_{speed} = \begin{cases} \frac{1}{2}(\hat{v} - v)^2 & \text{for } |\hat{v} - v| \le 1 \\ |\hat{v} - v| - \frac{1}{2} & \text{otherwise} \end{cases}$$
2.  **Uncertainty Loss ($L_{uncertainty}$)**: Heteroscedastic regression log-variance optimization:
    $$L_{uncertainty} = \frac{1}{2} \left[ e^{-s} (v - \hat{v})^2 + s \right]$$
    where $s = \log \sigma^2$.
3.  **Event Classification Loss ($L_{event}$)**: Cross-entropy loss mapping motion logits to the 5 targets.

---

## 5. Adaptive Model Controller

Fuses predictions from the three models to output a calibrated prediction.

### 5.1 Uncertainty-Weighted Fusion
Fuses candidate speed estimates ($\mu_i$) using weights ($w_i$) defined as inverse variances:
$$w_i = \frac{1}{\sigma_i^2 + \epsilon}$$
$$\bar{v} = \frac{\sum_{i=1}^3 w_i \mu_i}{\sum_{i=1}^3 w_i}, \quad \sigma_{fused}^2 = \frac{1}{\sum_{i=1}^3 w_i}$$

### 5.2 Model Disagreement Correction
If predictions disagree, the output uncertainty must increase. Disagreement is calculated as prediction variance across candidates and added to the fused variance:
$$\sigma_{adjusted}^2 = \sigma_{fused}^2 + \frac{1}{3}\sum_{i=1}^3 (\mu_i - \bar{v})^2$$

This adjusted variance is used to compute the **calibrated confidence**:
$$\text{confidence} = \frac{1}{1 + \sigma_{adjusted}^2} \in [0, 1]$$

---

## 6. Verification and Boundary Rules

To verify robustness and prevent data leakage, the following rules are enforced:
- **No Future Leakage**: Verified via a causality unit test ([test_causality.py](file:///c:/Users/surwe/Project/INS/model/tests/test_causality.py)) showing that changing elements in the second half of the time series does not alter the feature representations computed at previous steps.
- **Chronological Split**: Trajectory windows are split chronologically (70% train, 15% val, 15% test) to prevent identical overlapping sliding windows from appearing in both train and validation/test folds.
- **No V-M Dependency**: Deployed models receive only preprocessed $S-M$ data at inference. V-M targets are strictly used offline during development.
