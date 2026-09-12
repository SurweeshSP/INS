# Two-Model Adaptive Smartphone INS (6-Channel Focus)

This repository implements a lightweight, purely smartphone-based Inertial Navigation System (INS) estimator. The models are designed to replace GPS during prolonged outages by estimating instantaneous speed, motion state, and uncertainty using only **6-Channel (6C)** smartphone inertial sensors.

---

## 1. Data Preprocessing Pipeline (S-M & V-M Datasets)

The AI models are trained exclusively on smartphone data, while the high-precision vehicle data is used strictly for ground-truth labeling.

### Datasets
*   **S-M (Smartphone Dataset):** Raw IMU data from the phone's internal sensors.
    *   **Variables Used:** `Accelerometer X, Y, Z`, `Gyroscope Yaw, Pitch, Roll`, and `Gravity X, Y, Z`.
*   **V-M (Vehicle Dataset):** High-precision ground-truth data from the OBD-II port and professional INS units.
    *   **Variables Used:** `Velocity (km/h)`, `Longitudinal Acceleration (g)`, and `Motion Labels` (Braking, Accelerating, Turning, Stopped).

### Pipeline Execution (`preprocess_vehicle.py`)
1. **Temporal Alignment:**
   - The S-M and V-M timeseries are misaligned. We use cross-correlation (`find_optimal_shift`) between the smartphone's linear acceleration and the vehicle's OBD longitudinal acceleration to perfectly synchronize the data streams.
2. **Vehicle Frame Rotation:**
   - The smartphone can be placed in any orientation. We construct a 3x3 rotation matrix using:
     - **Z-axis:** Normalized average Gravity vector.
     - **X-axis:** The projection of smartphone linear acceleration that highly correlates with OBD longitudinal acceleration.
     - **Y-axis:** The cross product of Z and X.
   - The raw S-M inputs are then rotated into a fixed vehicle frame.
3. **6-Channel Extraction:**
   - For the 6C experiments, the models ingest purely the rotated linear acceleration ($Ax_v, Ay_v, Az_v$) and rotated gyroscope ($Gx_v, Gy_v, Gz_v$) data.
4. **Windowing:**
   - The aligned data is chunked into sliding windows (lengths $L = 20, 40, 60, 100$) using a strict temporal stride.

---

## 2. The Two-Model Architecture

Instead of relying on a monolithic network, we split the architecture into two specialized models deployed alongside an adaptive controller.

### Model A: ResNet1D + Multi-Head Attention (High Accuracy)
Designed to minimize positional drift over long outages by capturing deep contextual dependencies.
*   **Stem:** Causal Conv1D mapping 6 input channels to 32 dimensions.
*   **Residual Blocks:** Two sequential 1D Residual Blocks with channel progression `32 → 48 → 64`.
*   **Attention:** 2-Head Self-Attention layer (`embed_dim=64`) to capture long-term inertial dependencies.
*   **Characteristics:** Higher parameter count, higher latency, optimized for extremely low speed MAE and robust drift protection.

### Model B: Adaptive Dilated TCN (Ultra-Low Latency)
Designed as the default runtime workhorse, heavily optimized for execution on low-power smartphone edge chips.
*   **Stem:** 1x1 Pointwise Projection mapping 6 channels to 16 dimensions.
*   **Causal CNN:** Depthwise causal convolutions grouped for computational efficiency.
*   **Dilated TCN Blocks:** Temporal Convolutional Networks with progressive dilations ($d=1, d=2, d=4$) to exponentially increase the receptive field without adding parameters.
*   **Characteristics:** $< 15k$ parameters, $< 5ms$ latency, minimal memory footprint.

### Multi-Task Head Contract
Both models share the exact same output signature, predicting:
1. `speed_mps`: Instantaneous speed (Huber Loss).
2. `speed_variance`: Aleatoric uncertainty mapping for UKF integration (Gaussian NLL).
3. `motion_logits`: Classification of motion state (CrossEntropy Loss).

---

## 3. Evaluation & Graphing

The evaluation pipeline (`evaluator.py` & `plot_generator.py`) generates a comprehensive set of metrics and visualizations to guarantee navigation safety.

### Key Visualizations
*   **Speed vs. Ground Truth:** A temporal plot tracking predicted speed (m/s) against the V-M ground-truth velocity.
*   **Speed Residual Error:** Tracking the $y_{pred} - y_{true}$ error distribution across the dataset to ensure zero-mean bias.
*   **Stationary False-Speed:** A critical histogram analyzing the predicted speed when the car is physically stopped ($< 0.5$ m/s). False speeds here compound catastrophically during dead reckoning.

### Zero-Velocity Update (ZUPT)
We utilize the multi-task `motion_logits` to execute physical constraints. If a model predicts $P(\text{STOPPED}) \ge 0.8$, the evaluator artificially snaps the predicted `speed_mps` to exactly $0.0$. This completely halts stationary drift integration.

### Metric Reporting
The models are ranked based on the **60-second Positional Drift**, representing the percentage of positional error accumulated during a 1-minute GPS blackout, alongside Speed MAE, $R^2$, and Motion Macro-F1 scores.

---

## 4. Runtime Adaptive Fusion (`inference_adaptive_fusion.py`)

At deployment, we do not run both models concurrently. Instead:
1. **Model B (TCN)** runs by default, preserving battery life and meeting strict $<10ms$ latency requirements.
2. The controller monitors Model B's `speed_variance` (uncertainty) and `motion_class` outputs.
3. If uncertainty crosses a calibrated threshold (e.g., erratic driving) or a sharp motion transition occurs, the controller seamlessly activates **Model A (Attention)** for a 2-second cooldown window to maintain precision, before dropping back to Model B.
