# Lightweight AI/ML Smartphone INS Navigation Models

## GNSS-Aided Training Dataset: S-M + V-M

This repository develops and compares **three lightweight deep-learning models** for smartphone-based vehicle-speed estimation as an AI/ML component of an **Inertial Navigation System (INS)**.

The model is designed for the following deployment pipeline:

```text
S-M (Smartphone)
      │
      ▼
Sensor Preprocessing
      │
      ▼
Vehicle-frame IMU
      │
      ▼
20 × 6 temporal window
      │
      ▼
┌───────────────────────────────┐
│        Lightweight ML         │
│                               │
│  1. Bi-LSTM → Slim GRU        │
│  2. Transformer → Causal CNN  │
│  3. Wavelet → LSTM             │
└───────────────┬───────────────┘
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
      Speed   Variance  Motion
        │       │        │
        └───────┼────────┘
                ▼
             JSON
                │
                ▼
          INS / Dead Reckoning
```

During training, the V-M dataset supplies the reference vehicle state:

```text
V-M
 │
 ▼
Reference Preprocessing
 │
 ├── Vehicle velocity
 ├── GNSS/reference information
 ├── Wheel speeds
 └── Vehicle dynamics
 │
 ▼
Target Y + Motion Labels
 │
 ▼
Supervised / Distillation Loss
```

The **deployed model does not require V-M data**.

---

# 1. Project Objective

The primary learning problem is:

\[
\hat{v}_t = f_\theta(X_t)
\]

where:

- \(X_t\) = smartphone IMU sequence
- \(f_\theta\) = lightweight neural network
- \(\hat{v}_t\) = estimated vehicle speed in m/s

The model additionally estimates:

\[
(\hat v_t,\hat\sigma_t^2,\hat c_t)
\]

where:

- \(\hat v_t\) = speed
- \(\hat\sigma_t^2\) = predicted speed variance
- \(\hat c_t\) = motion-class probability distribution

The application layer converts these outputs into:

```json
{
  "timestamp": 12.40,
  "speed_mps": 13.42,
  "speed_variance": 0.35,
  "confidence": 0.91,
  "motion_class": "NORMAL"
}
```

The final research objective is not only speed estimation. The predicted velocity should be evaluated as an **INS aiding signal during GNSS-denied/degraded periods**.

---

# 2. Actual Dataset Used

This README is updated for the uploaded:

```text
S-M(2).csv
V-M(2).csv
```

## 2.1 S-M Dataset

The S-M dataset contains:

```text
105,974 rows
24 columns
```

Relevant smartphone sensor fields include:

```text
GPS LATITUDE
GPS LONGITUDE
GPS ALTITUDE
GPS SPEED
GPS ACCURACY
GPS ORIENTATION
GPS SATELLITES IN RANGE
TIME SINCE START
DATE
ACCELEROMETER X
ACCELEROMETER Y
ACCELEROMETER Z
GRAVITY X
GRAVITY Y
GRAVITY Z
GYROSCOPE Yaw
GYROSCOPE Pitch
GYROSCOPE Roll
MAGNETIC FIELD X
MAGNETIC FIELD Y
MAGNETIC FIELD Z
ORIENTATION Yaw
ORIENTATION Pitch
ORIENTATION Roll
```

### ML input channels

For the smartphone-only estimator, the six primary ML channels are:

```text
Accelerometer X
Accelerometer Y
Accelerometer Z
Gyroscope X/Yaw
Gyroscope Y/Pitch
Gyroscope Z/Roll
```

The smartphone GPS fields are **not used as model inputs** when evaluating a pure IMU-based estimator.

They can instead be retained for analysis and validation.

---

# 3. S-M Dataset Characteristics

The supplied S-M data contains:

```text
Rows:                  105,974
Columns:                    24
Accelerometer samples:    3 axes
Gyroscope samples:        3 axes
Missing primary IMU:       0
```

The recorded accelerometer values are approximately:

```text
X: -39.65 to 18.24 m/s²
Y: -39.36 to 12.81 m/s²
Z: -19.58 to 59.45 m/s²
```

The gyroscope values are recorded in:

```text
rad/s
```

with observed ranges approximately:

```text
Yaw:   -9.81 to  5.87 rad/s
Pitch: -8.45 to 12.66 rad/s
Roll:  -3.85 to  9.72 rad/s
```

These ranges show why **sensor cleaning, outlier handling, gravity compensation, and normalization are required before ML training**.

---

# 4. V-M Dataset

The V-M dataset contains:

```text
105,974 rows
29 columns
```

Important fields include:

```text
No of GPS Satellites Available
Time Since Start of Day
Latitude
Longitude
Velocity
Heading
Height
Vertical Velocity
Sample Period
Steering Angle
Wheel Speed Front Left
Wheel Speed Front Right
Wheel Speed Rear Left
Wheel Speed Rear Right
Yaw Rate
Indicated Vehicle Speed
Indicated Longitudinal Acceleration
Indicated Lateral Acceleration
Handbrake
Gear Requested
Gear
Engine Speed
Coolant Temperature
Clutch Position
Brake Pressure
Brake Position
Battery Voltage
Air Temperature
Accelerator Pedal Position
```

---

# 5. V-M Reference Target

The primary supervised target is:

```text
Velocity (km/hr)
```

The observed target range in the supplied dataset is approximately:

```text
0.0 to 100.688 km/h
```

Convert to SI units:

\[
V_{m/s} = \frac{V_{km/h}}{3.6}
\]

Therefore:

```text
V-M Velocity
      │
      ▼
   km/h
      │
      ▼
   ÷ 3.6
      │
      ▼
   target_y
      │
      ▼
    m/s
```

Maximum observed reference velocity is approximately:

\[
100.688/3.6 \approx 27.97\;m/s
\]

---

# 6. V-M Preprocessing

V-M is also preprocessed, but its preprocessing is different from S-M.

## S-M preprocessing

```text
Calibration
     ↓
Gravity correction
     ↓
Frame transformation
     ↓
Filtering
     ↓
Normalization
```

## V-M preprocessing

```text
Timestamp validation
       ↓
Missing-value check
       ↓
Outlier/reference-quality check
       ↓
GNSS / vehicle / wheel consistency
       ↓
Velocity unit conversion
       ↓
Motion-label generation
       ↓
Training target
```

V-M is therefore a **reference/target pipeline**, not a second feature stream for the deployed model.

---

# 7. V-M Reference Validation

Vehicle velocity should be checked against independent vehicle signals.

The supplied V-M dataset provides four wheel-speed channels:

```text
Front Left
Front Right
Rear Left
Rear Right
```

It also provides:

```text
Yaw Rate
Longitudinal Acceleration
Lateral Acceleration
Indicated Vehicle Speed
GNSS information
```

These signals can be used to detect suspicious reference samples.

Example:

```text
Vehicle Velocity
       │
       ├────────► Primary target
       │
       ├── Wheel speeds ───────► consistency check
       │
       ├── GNSS information ───► reference check
       │
       └── Vehicle dynamics ───► motion-label check
```

A large disagreement does not automatically mean that one signal is wrong. Such samples should be **flagged and investigated**, rather than silently discarded.

---

# 8. Timestamp Alignment

The two datasets both contain approximately 10 Hz temporal information.

### S-M

The `TIME SINCE START (ms)` field is present, and the timestamped date field has approximately 100 ms sample spacing.

### V-M

The `Time Since Start of Day (seconds)` field has an approximately 0.1 s sample period.

The supplied files contain the same number of rows:

```text
S-M = 105,974
V-M = 105,974
```

However:

> **Equal row counts do not by themselves prove perfect temporal synchronization.**

The timestamp/date fields must therefore be checked before training.

The S-M date timestamps begin around:

```text
2019-09-07 09:13:29.506
```

while V-M starts around:

```text
29608.4 seconds
```

which corresponds to approximately 08:13:28.4 when interpreted as seconds from midnight. This indicates an approximately one-hour clock offset, with a small additional alignment difference that should be explicitly handled by the synchronization module.

Therefore the implementation should align using timestamps rather than assuming:

```text
S-M row i == V-M row i
```

without verification.

---

# 9. Correct Dataset Synchronization

Recommended procedure:

```text
S-M timestamp
      │
      ▼
Convert to absolute/relative seconds
      │
      ▼
Clock-offset estimation
      │
      ▼
Timestamp matching/interpolation
      │
      ▼
Aligned S-M + V-M
```

After alignment:

```text
S-M(t)
  │
  ├── IMU features ─────────► X(t)
  │
  └─────────────────────────┐
                            │
V-M(t)                      │
  │                         │
  └── Vehicle velocity ───► Y(t)
                            │
                            ▼
                     Training pair
                      (X(t), Y(t))
```

---

# 10. Smartphone Feature Construction

After synchronization, construct:

\[
S_t =
[a_x,a_y,a_z,\omega_x,\omega_y,\omega_z]
\]

## Step 1 — Calibration

\[
a_{cal}=a_{raw}-b_a
\]

\[
\omega_{cal}=\omega_{raw}-b_g
\]

## Step 2 — Gravity correction

The accelerometer contains both vehicle acceleration and gravity.

Conceptually:

\[
a_{motion}=a_{raw}-g_{estimated}
\]

The S-M file already contains gravity-axis measurements, which can be used as part of the gravity/orientation processing pipeline.

## Step 3 — Vehicle-frame transformation

Transform:

```text
Smartphone frame
       ↓
Vehicle frame
```

so that:

```text
X → longitudinal
Y → lateral
Z → vertical
```

The exact rotation should be determined from the smartphone mounting/orientation procedure rather than assumed.

## Step 4 — Filtering

Apply lightweight filtering to suppress:

```text
sensor spikes
high-frequency vibration
measurement noise
```

## Step 5 — Normalization

Calculate training-set statistics:

\[
x'=\frac{x-\mu}{\sigma}
\]

Use the **same training mean/std** during validation, testing, and deployment.

---

# 11. 20 × 6 Temporal Window

The final input is:

```text
20 samples × 6 IMU channels
```

Therefore:

\[
X\in R^{20\times6}
\]

At approximately 10 Hz:

```text
20 samples
≈
2 seconds
```

Sliding windows:

```text
Window 1 → samples 1–20
Window 2 → samples 2–21
Window 3 → samples 3–22
...
```

Each window receives one reference speed target associated with its prediction timestamp.

---

# 12. Target Construction

For every aligned prediction time:

```text
V-M Velocity
     │
     ▼
 km/h
     │
     ▼
  / 3.6
     │
     ▼
m/s
     │
     ▼
Y_speed
```

Example:

```text
V-M:
Velocity = 48.312 km/h

Target:
48.312 / 3.6
≈ 13.42 m/s
```

---

# 13. Motion-Class Target

The three models also produce:

```text
motion_class
```

Recommended classes:

```text
STOPPED
ACCELERATING
DECELERATING
TURNING
NORMAL
```

Generate labels from V-M vehicle dynamics using:

```text
Vehicle velocity
Longitudinal acceleration
Lateral acceleration
Yaw rate
Steering angle
```

Initial rule structure:

```text
IF speed < stopped_threshold
    → STOPPED

ELSE IF turning condition
    → TURNING

ELSE IF longitudinal acceleration > accel_threshold
    → ACCELERATING

ELSE IF longitudinal acceleration < decel_threshold
    → DECELERATING

ELSE
    → NORMAL
```

The thresholds must be selected from the actual training distribution and validated on held-out drives.

---

# 14. Multi-Task Learning Objective

Each model predicts three quantities:

\[
\hat y =
[
\hat v,
\log\hat\sigma^2,
\hat p_{motion}
]
\]

where:

```text
v              → speed
log variance   → uncertainty
p_motion       → motion classification
```

The overall loss is:

\[
L =
\lambda_vL_{speed}
+
\lambda_uL_{uncertainty}
+
\lambda_mL_{motion}
+
\lambda_dL_{distillation}
\]

The distillation term is used only for Models 1 and 2.

---

# 15. Speed Loss

Use Huber regression:

\[
L_{speed}=Huber(\hat v,v_{VM})
\]

Huber loss is preferable to relying exclusively on MSE because vehicle/GNSS reference data can contain occasional outliers or synchronization errors.

---

# 16. Uncertainty Prediction

The model predicts:

\[
s=\log(\sigma^2)
\]

and:

\[
\sigma^2=e^s
\]

Use heteroscedastic regression:

\[
L_{uncertainty}
=
\frac12
\left[
e^{-s}(y-\mu)^2+s
\right]
\]

This allows the model to express increased uncertainty when the smartphone IMU signal is less informative.

---

# 17. Confidence

Do not directly define:

```text
confidence = 1 - variance
```

because variance and confidence do not naturally share the same scale.

Instead:

```text
Predicted variance
       ↓
Validation-set calibration
       ↓
Calibrated confidence
```

Conceptually:

```text
Low uncertainty
      ↓
High confidence

High uncertainty
      ↓
Low confidence
```

---

# 18. Model 1 — Bi-LSTM Teacher → Slim GRU

## Purpose

Optimize for:

```text
Very low parameter count
Low memory
CPU inference
Real-time operation
```

Architecture:

```text
S-M
 │
 ▼
20 × 6
 │
 ▼
Bi-LSTM Teacher
 │
 ▼
Feature representation
 │
 ▼
Slim GRU Student
 │
 ├── Speed
 ├── Variance
 └── Motion
```

### Teacher

```text
Input       20 × 6
Bi-LSTM     64
Bi-LSTM     32
Dense       32
```

### Student

```text
Input       20 × 6
GRU         32
Dense       16
Output heads
```

### Distillation

\[
L_{distill}=MSE(h_T,h_S)
\]

Total:

\[
L=
L_{speed}
+
\lambda_uL_{uncertainty}
+
\lambda_mL_{motion}
+
\lambda_dL_{distill}
\]

### Quantization

```text
FP32 student
     ↓
Representative calibration data
     ↓
INT8 PTQ
     ↓
Mobile deployment
```

---

# 19. Model 2 — Transformer Teacher → Causal 1D-CNN

## Purpose

Optimize for:

```text
Temporal feature extraction
High parallelism
Very low inference latency
Efficient convolutional execution
```

Architecture:

```text
S-M
 │
 ▼
20 × 6
 │
 ▼
Transformer Teacher
 │
 ▼
Temporal representation
 │
 ▼
Causal 1D-CNN Student
 │
 ├── Speed
 ├── Variance
 └── Motion
```

### Teacher

```text
Input projection       32
Transformer encoder    2 layers
Representation         32
```

### Student

```text
Input                 20 × 6
Causal Conv1D         16 filters
Causal Conv1D         24 filters
Dense                 16
Output heads
```

### Distillation

For continuous speed:

\[
L_{distill}=MSE(V_S,V_T)
\]

Total:

\[
L=
L_{speed}
+
\lambda_uL_{uncertainty}
+
\lambda_mL_{motion}
+
\lambda_dL_{distill}
\]

KL divergence should only be used for an appropriate probability-distribution output, not directly for continuous speed regression.

### Quantization

```text
CNN Student
     ↓
FP16 PTQ
     ↓
Mobile inference backend
```

Causality must be preserved so that the model cannot use future samples.

---

# 20. Model 3 — Wavelet → LSTM

## Purpose

Optimize for:

```text
Multi-scale temporal features
Road-vibration robustness
Sensor-noise separation
High-precision speed estimation
```

Architecture:

```text
S-M
 │
 ▼
20 × 6
 │
 ▼
Wavelet decomposition
 │
 ├── Low frequency
 ├── Mid frequency
 └── High frequency
 │
 ▼
LSTM
 │
 ▼
Dense
 │
 ├── Speed
 ├── Variance
 └── Motion
```

The wavelet stage should remain lightweight.

A practical design is:

```text
6 IMU channels
×
2 selected wavelet components
≈
12 feature channels
```

Then:

```text
20 × 12
   ↓
LSTM 32
   ↓
Dense 16
   ↓
Output heads
```

### Quantization

```text
FP32 model
     ↓
Fake INT8 quantization
     ↓
QAT
     ↓
INT8 deployment
```

---

# 21. Common Output Interface

All three models must expose the same application-level interface:

```text
Model Tensor
     │
     ├── speed
     ├── log_variance
     └── motion probabilities
     │
     ▼
Post-processing
     │
     ├── variance = exp(log_variance)
     ├── calibrated confidence
     └── argmax motion
     │
     ▼
JSON
```

Output:

```json
{
  "timestamp": 12.40,
  "speed_mps": 13.42,
  "speed_variance": 0.35,
  "confidence": 0.91,
  "motion_class": "NORMAL"
}
```

The JSON serialization is performed by the application layer, **not by the neural network**.

---

# 22. Data Leakage Prevention

The following V-M fields must not be model inputs for the smartphone-only speed estimator:

```text
Velocity
Wheel speeds
GNSS velocity
Vehicle acceleration
Yaw rate
Indicated vehicle speed
```

They are training references or validation signals.

Correct:

```text
S-M
 │
 ▼
ML Model
 │
 ▼
Prediction
 ▲
 │
Loss
 │
V-M
 │
 ▼
Reference target
```

Incorrect:

```text
S-M + V-M vehicle speed
        │
        ▼
       ML
        │
        ▼
Predicted vehicle speed
```

The second design leaks the target.

---

# 23. Train/Validation/Test Split

Do not randomly split overlapping windows.

Because:

```text
Window 1 = samples 1–20
Window 2 = samples 2–21
Window 3 = samples 3–22
```

random splitting can place almost identical windows in train and test.

Preferred:

```text
Drive / Trip A → Training
Drive / Trip B → Training
Drive / Trip C → Validation
Drive / Trip D → Testing
```

If multiple vehicles are available:

```text
Vehicle A → Training
Vehicle B → Validation
Vehicle C → Testing
```

This provides a much stronger generalization test.

---

# 24. Model Parameter and Latency Targets

These are **engineering targets**, not measured results.

| Metric | Bi-LSTM → GRU | Transformer → Causal CNN | Wavelet → LSTM |
|---|---:|---:|---:|
| Deployable network | Slim GRU | Causal CNN | LSTM |
| Quantization | INT8 PTQ | FP16 PTQ | INT8 QAT |
| Target parameters | ~6K–10K | ~15K–30K | ~10K–30K |
| Target weight storage | ~10–40 KB | ~30–120 KB | ~10–60 KB |
| Target latency | <2–3 ms | ~1–2 ms | ~2–5 ms |
| CPU | Excellent | Excellent | Good |
| GPU | Good | Excellent | Good |
| NPU/DSP | Good | Good | Excellent |
| Main advantage | Tiny recurrent model | Parallel execution | Multi-scale robustness |

**Actual latency, memory, and model size must be measured on the target smartphone.**

---

# 25. Latency Benchmark

For every exported model:

```text
Load model
    ↓
Warm-up 100–500 runs
    ↓
Run ≥1,000 inference windows
    ↓
Measure
    ├── Mean
    ├── P50
    ├── P95
    └── P99
```

Report:

```text
Mean latency
P50 latency
P95 latency
P99 latency
```

The most important real-time requirement is the **P95/P99 latency**, not only the average.

---

# 26. Model Performance Metrics

## Speed

```text
MAE
RMSE
R²
Maximum absolute error
```

## Motion classification

```text
Accuracy
Precision
Recall
F1-score
Confusion matrix
```

## Uncertainty

```text
Negative Log-Likelihood
Calibration error
Prediction interval coverage
Uncertainty-error correlation
```

## Deployment

```text
Parameter count
Model file size
Peak RAM
Mean latency
P95 latency
P99 latency
CPU/GPU/NPU utilization
Energy per inference
Thermal behavior
```

---

# 27. INS-Level Performance

The final experiment must compare:

```text
A. INS only

B. ML velocity estimator

C. INS + ML velocity aiding
```

During a simulated GNSS outage:

```text
GNSS available
      │
      ▼
GNSS outage
      │
      ├── 10 s
      ├── 30 s
      ├── 60 s
      └── 300 s
      │
      ▼
Navigation error
```

Measure:

```text
Position drift
Velocity error
Heading error
Trajectory deviation
GNSS-recovery error
```

The final model should be selected based on **navigation improvement + accuracy + uncertainty + latency + model size**, rather than speed RMSE alone.

---

# 28. Complete Training Pipeline

```text
                  S-M(2).csv
                      │
                      ▼
             Smartphone preprocessing
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
      Calibration  Gravity     Frame
                   removal    transform
          │           │           │
          └───────────┼───────────┘
                      ▼
                  Filtering
                      │
                      ▼
                Normalization
                      │
                      ▼
                   20 × 6
                      │
                      ▼
              ┌───────┴────────┐
              │                │
              │                │
              ▼                │
       Lightweight ML          │
              │                │
       ┌──────┼───────┐        │
       ▼      ▼       ▼        │
     Speed Variance Motion      │
              │                │
              └──────┬─────────┘
                     │
                     ▼
                Predictions
                     │
                     │ Loss
                     ▲
                     │
              V-M reference
                     ▲
                     │
               V-M(2).csv
                     │
                     ▼
              Reference preprocessing
                     │
          ┌──────────┼───────────┐
          ▼          ▼           ▼
       Velocity   Wheel speed   Dynamics
          │          │           │
          ▼          ▼           ▼
       Target Y   Validation   Motion labels
```

---

# 29. Repository Structure

```text
smartphone-ins-ml/
│
├── README.md
│
├── data/
│   ├── raw/
│   │   ├── S-M(2).csv
│   │   └── V-M(2).csv
│   │
│   └── processed/
│       ├── train.npz
│       ├── validation.npz
│       └── test.npz
│
├── preprocessing/
│   ├── smartphone.py
│   ├── vehicle_reference.py
│   ├── synchronization.py
│   ├── gravity.py
│   └── coordinate_transform.py
│
├── features/
│   ├── imu_features.py
│   └── wavelet_features.py
│
├── models/
│   │
│   ├── model1_bilstm_gru/
│   │   ├── architecture/
│   │   │   ├── teacher.py
│   │   │   ├── student.py
│   │   │   └── heads.py
│   │   │
│   │   ├── training/
│   │   │   ├── train_teacher.py
│   │   │   ├── train_student.py
│   │   │   └── distillation.py
│   │   │
│   │   ├── quantization/
│   │   │   └── int8_ptq.py
│   │   │
│   │   ├── evaluation/
│   │   │   ├── speed_metrics.py
│   │   │   ├── motion_metrics.py
│   │   │   ├── uncertainty.py
│   │   │   ├── latency.py
│   │   │   └── model_size.py
│   │   │
│   │   ├── checkpoints/
│   │   └── exports/
│   │
│   ├── model2_transformer_cnn/
│   │   ├── architecture/
│   │   │   ├── teacher.py
│   │   │   ├── student.py
│   │   │   └── heads.py
│   │   │
│   │   ├── training/
│   │   │   ├── train_teacher.py
│   │   │   ├── train_student.py
│   │   │   └── distillation.py
│   │   │
│   │   ├── quantization/
│   │   │   └── fp16_ptq.py
│   │   │
│   │   ├── evaluation/
│   │   │   ├── speed_metrics.py
│   │   │   ├── motion_metrics.py
│   │   │   ├── uncertainty.py
│   │   │   ├── latency.py
│   │   │   └── model_size.py
│   │   │
│   │   ├── checkpoints/
│   │   └── exports/
│   │
│   └── model3_wavelet_lstm/
│       ├── architecture/
│       │   ├── wavelet.py
│       │   ├── model.py
│       │   └── heads.py
│       │
│       ├── training/
│       │   └── train.py
│       │
│       ├── quantization/
│       │   └── int8_qat.py
│       │
│       ├── evaluation/
│       │   ├── speed_metrics.py
│       │   ├── motion_metrics.py
│       │   ├── uncertainty.py
│       │   ├── latency.py
│       │   └── model_size.py
│       │
│       ├── checkpoints/
│       └── exports/
│
├── ins/
│   ├── dead_reckoning.py
│   └── fusion.py
│
├── deployment/
│   ├── android/
│   └── ios/
│
└── outputs/
    ├── predictions/
    ├── metrics/
    └── benchmarks/
```

---

# 30. Model Development Order

Build the models in this order.

## Phase 1 — Data validation

```text
S-M(2).csv
V-M(2).csv
     ↓
Timestamp synchronization
     ↓
Reference validation
     ↓
20 × 6 dataset
```

## Phase 2 — Baseline

```text
20 × 6
 ↓
GRU(32)
 ↓
Speed + variance + motion
```

This determines whether smartphone IMU alone contains enough information for the task.

## Phase 3 — Model 1

```text
Bi-LSTM Teacher
       ↓
Slim GRU
       ↓
INT8 PTQ
```

## Phase 4 — Model 2

```text
Transformer Teacher
       ↓
Causal CNN
       ↓
FP16 PTQ
```

## Phase 5 — Model 3

```text
Wavelet
   ↓
LSTM
   ↓
INT8 QAT
```

## Phase 6 — INS integration

```text
Predicted velocity
       ↓
INS fusion
       ↓
Navigation state
```

---

# 31. Final Comparison

The final experiment should produce a table similar to:

| Metric | Model 1 | Model 2 | Model 3 |
|---|---:|---:|---:|
| Parameters | Measured | Measured | Measured |
| FP32 size | Measured | Measured | Measured |
| Quantized size | Measured | Measured | Measured |
| Speed MAE | Measured | Measured | Measured |
| Speed RMSE | Measured | Measured | Measured |
| R² | Measured | Measured | Measured |
| Motion F1 | Measured | Measured | Measured |
| Uncertainty calibration | Measured | Measured | Measured |
| Mean latency | Measured | Measured | Measured |
| P95 latency | Measured | Measured | Measured |
| P99 latency | Measured | Measured | Measured |
| Peak RAM | Measured | Measured | Measured |
| 60 s GNSS-outage drift | Measured | Measured | Measured |
| 300 s GNSS-outage drift | Measured | Measured | Measured |

This table should contain **measured results only**. The latency and parameter numbers in the architecture sections are design targets, not claimed experimental results.

---

# 32. Final System

```text
                         S-M(2).csv
                              │
                              ▼
                    Smartphone IMU
                              │
                              ▼
                    Sensor preprocessing
                              │
                              ▼
                    Vehicle-frame IMU
                              │
                              ▼
                           20 × 6
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
       BiLSTM → GRU    Transformer → CNN   Wavelet → LSTM
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                       Multi-task output
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
                  Speed    Variance   Motion
                    │         │         │
                    └─────────┼─────────┘
                              ▼
                            JSON
                              │
                              ▼
                         INS Fusion
                              │
                              ▼
                  GNSS-denied navigation
                              │
                              ▼
                    Position / Velocity /
                         Heading
```

Training reference:

```text
                         V-M(2).csv
                              │
                              ▼
                    V-M preprocessing
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
         Vehicle speed    Wheel speeds   Vehicle dynamics
               │              │              │
               ▼              ▼              ▼
            Target Y      Validation    Motion labels
               │
               ▼
          Supervised loss
               │
               ▼
             ML model
```

---

# 33. Core Research Hypothesis

The complete experiment tests:

> **Can a lightweight smartphone-only temporal model estimate vehicle speed, uncertainty, and motion state from a 20 × 6 IMU window with sufficiently low latency and model size to provide useful velocity aiding for INS during GNSS-denied navigation?**

The three architectures provide three different optimization strategies:

```text
MODEL 1
Bi-LSTM → Slim GRU
        ↓
Tiny + CPU + low memory


MODEL 2
Transformer → Causal CNN
        ↓
Parallel + low latency


MODEL 3
Wavelet → LSTM
        ↓
Multi-scale + robustness
```

The final winner is the model that provides the best **INS-level navigation improvement under the required latency, memory, uncertainty, and power constraints**.
