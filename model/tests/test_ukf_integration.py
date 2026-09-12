import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.evaluation import INS_UKF_Simulator

def test_ukf_integration():
    print("Running test_ukf_integration...")
    
    # Initialize simulator
    sim = INS_UKF_Simulator(dt=0.1, q_noise=0.01, r_base=0.1)
    
    # 1. Propagate with constant acceleration 2.0 m/s^2
    # At start: v = 0, x = 0
    sim.propagate(ax_vehicle=2.0)
    
    # v = 0 + 2.0 * 0.1 = 0.2
    # x = 0 + 0.2 * 0.1 + 0.5 * 2.0 * 0.01 = 0.03
    assert abs(sim.v - 0.2) < 1e-5
    assert abs(sim.x - 0.03) < 1e-5
    
    # 2. Update simulator using virtual ML speed of 0.3 m/s
    sim.update_ml_velocity(speed_pred=0.3, variance_pred=0.05, confidence_pred=0.9)
    
    # Corrected velocity should be between INS propagated (0.2) and ML measurement (0.3)
    assert 0.2 < sim.v < 0.3
    print(f"Corrected velocity: {sim.v:.4f} m/s (between 0.2 and 0.3)")
    
    print("test_ukf_integration passed successfully!")

if __name__ == '__main__':
    test_ukf_integration()
