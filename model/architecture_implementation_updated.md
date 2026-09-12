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
    Model 1              Model 2              Model 3              Model 4
    Bi-LSTM → GRU        Transformer → CNN   Wavelet → LSTM       ResNet1D + Attention
    (Feature Distill)    (Response Distill)  (QAT)               (INT8 QAT / Distill)
       │                            │                            │
       ▼                            ▼                            ▼
    INT8 PTQ                     FP16 PTQ                     INT8 PTQ
       │                            │                            │
       └────────────────────────────┼────────────────────────────┘
                                    ▼
                         Adaptive Controller / Model Selector
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

> **Gating condition — read before implementing this section.**
> Sections 3–5 (model training, distillation, fusion, adaptive controller) must not begin
> until the §6 diagnostic ablations (input channels, temporal window, filter type) have been
> run on the *existing* pipeline and confirm there is no labeling, target-alignment, or
> frame-alignment defect. The current baseline shows identical Motion Accuracy (62.7%) and
> Macro-F1 (48.3%) across three structurally different models plus their fusion output, and a
> negative Speed R² — a pattern far more consistent with a shared upstream bug (label
> encoding, class-imbalance handling, or unaligned sensor axes) than with three independent
> architectures converging on the same insufficient capacity. Building a four-model ensemble
> with distillation and adaptive fusion on top of an unverified pipeline risks compounding a
> foundational error rather than fixing it. Run §6's ablations first, on Model 2 (cheapest to
> iterate), confirm clean labels/targets and a meaningful accuracy jump from vehicle-frame
> alignment, and only then proceed to full training of Models 1–4 as specified below.

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

## 3.4 Model 4: ResNet1D + Multi-Head Self-Attention

Model 4 is the high-capacity reference architecture. It is designed to capture both local inertial patterns and longer temporal dependencies while remaining small enough for edge deployment.

### Input

The primary implementation uses the same smartphone-only feature contract as the other models:

```text
N × 9
```

with:

```text
Ax_vehicle
Ay_vehicle
Az_vehicle
GyroYaw_vehicle
GyroPitch_vehicle
GyroRoll_vehicle
Gravity_X
Gravity_Y
Gravity_Z
```

A 6-channel mode must also remain supported for ablation:

```text
N × 6
```

The recommended starting configuration is:

```text
Window length: 100 samples
Sampling rate: ~10 Hz
Temporal context: ~10 seconds
```

The shorter 20×9 configuration must also be benchmarked to determine the latency/accuracy trade-off.

This benchmark must be read together with the §6 required input and temporal ablations, not
in isolation: since Model 4's default config (100×9) changes *both* channel count and window
length relative to Models 1–3 (20×6) simultaneously, its raw accuracy number alone cannot tell
you whether a gain came from the extra 3 gravity channels, the 5x longer window, or the
architecture itself. Use the §6 ablation grid (6 vs 9 channels; 20/40/60/100-step windows) to
attribute Model 4's advantage to a specific factor before deciding what to port back into
Models 1–3.

### Architecture

```text
Input: N × 9
       │
       ▼
1D Conv Stem
9 → 32 or 64 channels
       │
       ▼
Residual Block 1
       │
       ▼
Residual Block 2
32/64 → 64/128 channels
       │
       ▼
Temporal Downsampling
       │
       ▼
Multi-Head Self-Attention
       │
       ▼
Residual Attention Connection
       │
       ▼
LayerNorm
       │
       ▼
Global Temporal Average Pooling
       │
       ▼
Dense 128 → 64
       │
       ├──────── Speed
       ├──────── Log Variance
       └──────── Motion
```

The residual blocks provide learned local temporal filtering while self-attention models longer-range relationships.

### Filtering role

The ResNet convolution layers act as learned temporal FIR-like filters:

```text
Conv filters
   ├── low-frequency vehicle dynamics
   ├── acceleration/braking transitions
   ├── vibration signatures
   └── cross-channel IMU relationships
```

The residual path prevents useful information from being lost through repeated filtering.

The attention layer is not intended to replace physical filtering. It learns which timesteps/channels are important for the current driving state.

### Model 4 Distillation

Model 4 is defined above as the accuracy-oriented reference model, explicitly **not** the
model auto-selected for deployment. Distillation below therefore activates under one
condition only: **if, after measuring FP32 Model 4 on-device, its latency/size makes it a
viable deployment candidate on its own (e.g. for budget-mode escalation in §5.2), and the team
decides to promote it to a deployable role.** If Model 4 remains reference-only, skip this
subsection — do not distill it by default.

Model 4 should use **Model 4 Teacher → Mobile ResNet Student distillation** only if the measured FP32 architecture is too large for the target device.

Teacher:

```text
ResNet1D + Attention
larger channel dimensions
more attention capacity
```

Student:

```text
Slim ResNet1D
fewer channels
fewer residual blocks
single lightweight attention block
```

Use:

```text
supervised multi-task loss
+
feature distillation
+
response distillation
```

For example:

\[
L_4 =
L_{task}
+
\lambda_f MSE(h_S,h_T)
+
\lambda_v MSE(\mu_S,\mu_T)
\]

The teacher is never deployed.

### Model 4 Quantization

Use **INT8 Quantization-Aware Training (QAT)** as the primary deployment path.

```text
FP32 ResNet Student
        │
        ▼
Fake Quantization
        │
        ▼
QAT
        │
        ▼
INT8 deployment model
```

If the target mobile accelerator provides a better FP16 path, FP16 PTQ may be evaluated as a secondary variant.

The implementation must compare:

```text
FP32
FP16
INT8 PTQ/QAT
```

for:

```text
accuracy
model size
latency
memory
```

### Target

Model 4 is the **accuracy-oriented reference model**, not the model that should automatically be selected for deployment.

Its target is:

```text
Motion accuracy ≥ 90%
Macro-F1 ≥ 0.90
Speed R² ≥ 0.90 target
P99 latency < 10 ms target
```

These are acceptance targets and must be measured on held-out data.


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

## 5. Adaptive Model Controller and Adaptive Learning

The four models are complementary rather than identical:

```text
Model 1: GRU
→ adaptive recurrent temporal filtering

Model 2: Causal CNN
→ learned causal FIR-like filtering

Model 3: Wavelet-LSTM
→ explicit multi-scale frequency decomposition

Model 4: ResNet1D + Attention
→ residual local filtering + long-range attention
```

The controller selects/weights models according to their estimated reliability.

```text
                         S-M IMU
                            │
                            ▼
                       Shared input
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
        Model 1           Model 2           Model 3
          │                 │                 │
          └─────────────────┬─────────────────┘
                            │
                            ▼
                         Model 4
                            │
                            ▼
                  Adaptive Controller
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
      uncertainty       disagreement       motion state
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                   Dynamic model weights
                            │
                            ▼
                  Fused speed/variance/event
```

### 5.1 Calibrated Uncertainty Fusion

Do not directly trust raw neural variance.

First calibrate each model on the validation set.

For model \(i\):

\[
\tilde{\sigma}_i^2 = Calibration(\sigma_i^2)
\]

Then calculate reliability:

\[
r_i =
\exp(-\alpha\tilde{\sigma}_i^2-\beta d_i)
\]

where \(d_i\) is disagreement between model \(i\) and the ensemble.

Normalize:

\[
w_i = \frac{r_i}{\sum_jr_j}
\]

and fuse:

\[
\hat v =
\sum_{i=1}^{4}w_i\mu_i
\]

The fused variance includes both model uncertainty and inter-model disagreement.

### 5.2 Adaptive Model Selection

The controller should be able to operate in two modes.

**Fusion mode:**

```text
run all four models
→ dynamically weight predictions
```

**Budget mode:**

```text
run the cheapest/highest-confidence model
→ activate another model only when uncertainty rises
```

Example:

```text
NORMAL_CRUISING
      ↓
Model 2 sufficient
      ↓
low latency


HIGH UNCERTAINTY
      ↓
activate Model 1 / Model 3


HIGH VIBRATION
      ↓
increase Model 3 weight


COMPLEX DYNAMICS
      ↓
increase Model 4 weight
```

This makes the system adaptive in both **accuracy** and **compute cost**.

### 5.3 Adaptive Learning Without V-M Runtime Dependency

The deployed model must not learn from V-M data during smartphone inference.

Adaptive learning is limited to model-side parameters that can safely be updated from the current S-M stream, such as:

```text
normalization statistics
confidence calibration
temporal event smoothing
model-selection weights
small gating parameters
```

The neural network's main weights remain frozen unless an explicitly controlled online-learning experiment is enabled.

The online controller receives only:

```text
S-M IMU
```

and its own predictions.

No V-M feature is required.

### 5.4 V-M Reference Role

V-M is used offline as a **reference/ground-truth source**, not as an inference feature.

```text
V-M
 │
 ├── reference speed
 ├── reference dynamics
 └── offline event labels
       │
       ▼
 evaluation / calibration / target generation
```

It must never enter:

```text
X_runtime
```

Therefore:

```text
X_runtime = S-M only
```

The important distinction is:

> V-M can define what the model should predict, but V-M measurements are never provided to the deployed model as input.

If the phrase "not training on V-M" is used operationally, interpret it as **no V-M feature channels or V-M sensor signals are used as model inputs**. Supervised target generation from the reference speed/event labels is still permitted and required for IO-VNBD evaluation.


## 6. Acceptance Criteria and Required Ablation Study

The implementation must not claim that the models achieve 90% accuracy until the held-out test set demonstrates it.

For every model, the acceptance report must include:

```text
Motion Accuracy   >= 0.90 target
Motion Macro-F1   >= 0.90 target
Speed R²          >= 0.90 target
Speed MAE         <= 1.0 m/s target
Drift error       <= 10% of distance travelled, over rolling 30-60s horizons (target)
P99 latency       < 10 ms target
Quantized size    < 100 KB target
```

If a model misses a target, report the measured result and continue optimization rather than fabricating or substituting a target value.

### Required drift-integration metric

Per-window metrics (Motion Accuracy, F1, MAE, R²) can all pass while the deployed system
still fails the actual SIH benchmark, because that benchmark is measured on **cumulative
position drift**, not per-window speed error. A model with zero-mean but *biased* errors
(e.g. consistently underestimating speed during braking) can look acceptable per-window and
still drift unacceptably once integrated over a real drive.

For every model and for the fused output, in addition to the per-window table above, report:

```text
1. Integrate predicted speed over rolling 30s and 60s windows on held-out trajectories.
2. Compare integrated distance against V-M ground-truth distance over the same span.
3. Report drift error as % of true distance travelled, and whether the error is
   systematically signed (biased) or zero-mean (random).
4. Report drift error separately during GNSS-denied simulated segments vs normal segments.
```

A model that passes the per-window table but shows biased or excessive drift error must not
be marked as meeting the acceptance criteria, even if every other row in the table is green.

### Required input ablation

Run the same models with:

```text
6 channels:
N × 6

9 channels:
N × 9
```

and compare the contribution of:

```text
Gravity_X
Gravity_Y
Gravity_Z
```

### Required temporal ablation

Compare:

```text
20 × 9
40 × 9
60 × 9
100 × 9
```

while keeping the train/validation/test split fixed.

This determines whether the accuracy gain comes from additional gravity/orientation information, longer temporal context, or both.

### Required filter ablation

Compare:

```text
minimal preprocessing + learned filtering
light conventional filtering + learned filtering
wavelet filtering + learned temporal filtering
```

The objective is to determine whether conventional filtering improves or removes useful vehicle-dynamics information.

### Required controller validation

§5.2's adaptive model-selection logic (escalate to Model 1/3 on high uncertainty, increase
Model 3 weight on high vibration, increase Model 4 weight on complex dynamics) is a set of
claims about *when* the controller should trust which model — and those claims must be
checked against outcomes, not just implemented as described:

```text
1. For each escalation trigger (uncertainty spike, vibration, complex dynamics), confirm on
   held-out data that escalating actually reduces speed/motion error versus not escalating,
   for a matched set of windows.
2. Confirm the controller's uncertainty/vibration/complexity signals correlate with periods
   of genuinely higher ground-truth error, not just with noisy or unfamiliar input.
3. Report the compute/latency saved by budget mode against the accuracy cost, if any,
   relative to always running fusion mode.
```

If escalation does not measurably reduce error, the trigger should be reconsidered rather
than kept on the strength of its intuitive justification.

## 7. Verification and Boundary Rules

To verify robustness and prevent data leakage, the following rules are enforced:
- **No Future Leakage**: Verified via a causality unit test ([test_causality.py](file:///c:/Users/surwe/Project/INS/model/tests/test_causality.py)) showing that changing elements in the second half of the time series does not alter the feature representations computed at previous steps.
- **Chronological Split**: Trajectory windows are split chronologically (70% train, 15% val, 15% test) to prevent identical overlapping sliding windows from appearing in both train and validation/test folds.
- **No V-M Dependency**: Deployed models receive only preprocessed $S-M$ data at inference. V-M targets are strictly used offline during development.
