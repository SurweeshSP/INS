import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.train_best_accuracy import compute_drift

def test_drift_evaluation():
    print("Running test_drift_evaluation...")
    
    # 1. Create a dummy true speed profile (constant 10 m/s)
    true_speeds = np.ones(1000) * 10.0
    
    # 2. Create predicted speed with 5% constant bias (9.5 m/s)
    pred_speeds = np.ones(1000) * 9.5
    
    # 3. Compute drift for 30s horizon (300 steps)
    drift30 = compute_drift(true_speeds, pred_speeds, horizon_steps=300)
    
    # The true travelled distance over 300 steps (30 seconds) is 30 * 10 = 300 m.
    # The predicted distance is 30 * 9.5 = 285 m.
    # Drift % = |285 - 300| / 300 * 100% = 5.0%
    print(f"Calculated 30s drift: {drift30:.2f}% (Expected: 5.00%)")
    assert abs(drift30 - 5.00) < 1e-5, f"Incorrect drift calculation: {drift30}"
    
    print("test_drift_evaluation passed successfully!")

if __name__ == '__main__':
    test_drift_evaluation()
