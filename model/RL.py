"""
HMM-RL Adaptive Map Matching
============================

Architecture:

GNSS + IMU
     |
     v
Drift Prediction
     |
     v
Augmented Navigation State
     |
     v
HMM Candidate Generation
     |
     v
RL Policy
     |
     +----> Transition adaptation
     +----> Emission adaptation
     +----> Map confidence
     |
     v
HMM Forward/Viterbi inference
     |
     v
Most probable road state
     |
     v
Map-matched position

This implementation is intentionally lightweight and interpretable.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import math
import random
import numpy as np


# ============================================================
# 1. DATA STRUCTURES
# ============================================================

@dataclass
class NavigationObservation:
    timestamp: float

    latitude: float
    longitude: float

    speed: float
    heading: float

    acceleration: float
    yaw_rate: float

    # Predicted by external drift model
    predicted_drift: float = 0.0
    predicted_velocity_reduction: float = 0.0
    drift_confidence: float = 0.0

    # GNSS quality
    gnss_accuracy: float = 5.0
    gnss_confidence: float = 1.0


@dataclass
class RoadCandidate:
    road_id: str

    latitude: float
    longitude: float

    road_heading: float
    road_speed: float

    distance: float
    road_type: str = "residential"


@dataclass
class HMMState:
    road_id: str
    probability: float = 0.0


# ============================================================
# 2. BASIC GEOMETRY
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Approximate distance between two latitude/longitude points.
    Returns metres.
    """

    R = 6371000.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))


def angular_difference(a, b):
    """
    Smallest difference between two headings.
    """

    diff = abs(a - b) % 360

    if diff > 180:
        diff = 360 - diff

    return diff


# ============================================================
# 3. DRIFT PREDICTION MODEL
# ============================================================

class DriftPredictionModel:
    """
    Lightweight analytical drift predictor.

    In the real application this can be replaced with:
        - LSTM
        - GRU
        - Transformer
        - XGBoost
        - temporal CNN
        - your trained drift model
    """

    def __init__(self):

        self.previous_speed = None
        self.accumulated_drift = 0.0

    def predict(
        self,
        speed,
        acceleration,
        yaw_rate,
        dt
    ):

        if self.previous_speed is None:
            self.previous_speed = speed

        velocity_reduction = max(
            0.0,
            self.previous_speed - speed
        )

        # Simple analytical drift accumulation
        drift_increment = (
            abs(acceleration) * 0.02
            + abs(yaw_rate) * 0.01
        ) * dt

        self.accumulated_drift += drift_increment

        # Confidence increases with observed motion consistency
        confidence = min(
            1.0,
            0.2 + self.accumulated_drift / 20.0
        )

        self.previous_speed = speed

        return {
            "predicted_drift": self.accumulated_drift,
            "predicted_velocity_reduction": velocity_reduction,
            "confidence": confidence
        }


# ============================================================
# 4. HMM-BASED MAP MATCHER
# ============================================================

class HMMMapMatcher:

    def __init__(self):

        self.states: List[HMMState] = []

        # Base parameters
        self.distance_sigma = 20.0
        self.heading_sigma = 30.0
        self.speed_sigma = 10.0

        self.transition_weight = 1.0
        self.emission_weight = 1.0

        self.previous_road = None

    # --------------------------------------------------------
    # EMISSION PROBABILITY
    # --------------------------------------------------------

    def emission_probability(
        self,
        observation: NavigationObservation,
        candidate: RoadCandidate
    ):

        # ------------------------------------------
        # Distance likelihood
        # ------------------------------------------

        distance_error = candidate.distance

        distance_prob = math.exp(
            -(distance_error ** 2)
            / (2 * self.distance_sigma ** 2)
        )

        # ------------------------------------------
        # Heading likelihood
        # ------------------------------------------

        heading_error = angular_difference(
            observation.heading,
            candidate.road_heading
        )

        heading_prob = math.exp(
            -(heading_error ** 2)
            / (2 * self.heading_sigma ** 2)
        )

        # ------------------------------------------
        # Speed likelihood
        # ------------------------------------------

        speed_error = abs(
            observation.speed -
            candidate.road_speed
        )

        speed_prob = math.exp(
            -(speed_error ** 2)
            / (2 * self.speed_sigma ** 2)
        )

        # ------------------------------------------
        # Drift awareness
        # ------------------------------------------

        drift_factor = 1.0 / (
            1.0 +
            observation.predicted_drift
        )

        probability = (
            distance_prob
            * heading_prob
            * speed_prob
            * drift_factor
        )

        return max(probability, 1e-12)

    # --------------------------------------------------------
    # TRANSITION PROBABILITY
    # --------------------------------------------------------

    def transition_probability(
        self,
        previous_candidate: RoadCandidate,
        current_candidate: RoadCandidate
    ):

        if previous_candidate.road_id == current_candidate.road_id:

            return 0.85

        distance = haversine_distance(
            previous_candidate.latitude,
            previous_candidate.longitude,
            current_candidate.latitude,
            current_candidate.longitude
        )

        # Nearby roads are more likely transitions
        probability = math.exp(
            -distance / 50.0
        )

        return max(probability, 1e-8)

    # --------------------------------------------------------
    # HMM INFERENCE
    # --------------------------------------------------------

    def infer(
        self,
        observation: NavigationObservation,
        candidates: List[RoadCandidate]
    ):

        if len(candidates) == 0:
            return None

        probabilities = []

        for candidate in candidates:

            emission = self.emission_probability(
                observation,
                candidate
            )

            # First observation
            if self.previous_road is None:

                transition = 1.0

            else:

                transition = self.transition_probability(
                    self.previous_road,
                    candidate
                )

            score = (
                emission ** self.emission_weight
                *
                transition ** self.transition_weight
            )

            probabilities.append(score)

        probabilities = np.asarray(probabilities)

        total = probabilities.sum()

        if total <= 0:
            probabilities[:] = 1.0 / len(probabilities)
        else:
            probabilities /= total

        best_index = int(
            np.argmax(probabilities)
        )

        best_candidate = candidates[best_index]

        self.previous_road = best_candidate

        return {
            "road": best_candidate,
            "probabilities": probabilities
        }


# ============================================================
# 5. RL AGENT
# ============================================================

class AdaptiveRLAgent:
    """
    Lightweight tabular RL controller.

    RL does NOT directly predict latitude/longitude.

    Instead, RL learns how strongly HMM should trust:
        - distance
        - heading
        - speed
        - transitions
        - map constraint

    This keeps the system interpretable.
    """

    def __init__(self):

        # Actions
        #
        # 0 = trust GNSS
        # 1 = trust HMM/map
        # 2 = increase heading importance
        # 3 = increase speed importance
        # 4 = increase transition importance

        self.actions = [
            "GNSS_TRUST",
            "MAP_TRUST",
            "HEADING_TRUST",
            "SPEED_TRUST",
            "TRANSITION_TRUST"
        ]

        self.q_table = {}

        self.learning_rate = 0.1
        self.gamma = 0.9

        self.epsilon = 0.10

    # --------------------------------------------------------
    # STATE DISCRETIZATION
    # --------------------------------------------------------

    def discretize_state(
        self,
        observation: NavigationObservation,
        map_error: float
    ):

        if observation.gnss_accuracy < 5:
            gnss_state = 0

        elif observation.gnss_accuracy < 15:
            gnss_state = 1

        else:
            gnss_state = 2

        if observation.predicted_drift < 2:
            drift_state = 0

        elif observation.predicted_drift < 10:
            drift_state = 1

        else:
            drift_state = 2

        if map_error < 5:
            map_state = 0

        elif map_error < 20:
            map_state = 1

        else:
            map_state = 2

        velocity_state = int(
            min(
                2,
                observation.predicted_velocity_reduction
                // 2
            )
        )

        return (
            gnss_state,
            drift_state,
            map_state,
            velocity_state
        )

    # --------------------------------------------------------
    # Q TABLE
    # --------------------------------------------------------

    def get_q_values(self, state):

        if state not in self.q_table:

            self.q_table[state] = np.zeros(
                len(self.actions)
            )

        return self.q_table[state]

    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    def choose_action(self, state):

        q_values = self.get_q_values(state)

        if random.random() < self.epsilon:

            return random.randrange(
                len(self.actions)
            )

        return int(
            np.argmax(q_values)
        )

    # --------------------------------------------------------
    # POLICY
    # --------------------------------------------------------

    def apply_action(
        self,
        action,
        hmm: HMMMapMatcher
    ):

        # Reset adaptive parameters

        hmm.transition_weight = 1.0
        hmm.emission_weight = 1.0

        if action == 0:

            # Trust GNSS
            hmm.distance_sigma = 30.0

        elif action == 1:

            # Trust map
            hmm.distance_sigma = 10.0
            hmm.transition_weight = 1.5

        elif action == 2:

            # Trust heading
            hmm.heading_sigma = 15.0

        elif action == 3:

            # Trust speed
            hmm.speed_sigma = 5.0

        elif action == 4:

            # Trust temporal continuity
            hmm.transition_weight = 2.0

    # --------------------------------------------------------
    # Q UPDATE
    # --------------------------------------------------------

    def update(
        self,
        state,
        action,
        reward,
        next_state
    ):

        q_values = self.get_q_values(state)

        next_q = self.get_q_values(
            next_state
        )

        td_target = (
            reward
            +
            self.gamma * np.max(next_q)
        )

        td_error = (
            td_target
            -
            q_values[action]
        )

        q_values[action] += (
            self.learning_rate
            * td_error
        )


# ============================================================
# 6. HMM + RL SYSTEM
# ============================================================

class HMMRLNavigationSystem:

    def __init__(self):

        self.drift_model = (
            DriftPredictionModel()
        )

        self.hmm = HMMMapMatcher()

        self.rl = AdaptiveRLAgent()

        self.previous_position = None
        self.previous_state = None
        self.previous_error = None

    # --------------------------------------------------------
    # PROCESS ONE OBSERVATION
    # --------------------------------------------------------

    def process(
        self,
        observation: NavigationObservation,
        candidates: List[RoadCandidate],
        dt: float = 0.1
    ):

        # ====================================================
        # STEP 1
        # DRIFT PREDICTION
        # ====================================================

        drift = self.drift_model.predict(
            speed=observation.speed,
            acceleration=observation.acceleration,
            yaw_rate=observation.yaw_rate,
            dt=dt
        )

        observation.predicted_drift = (
            drift["predicted_drift"]
        )

        observation.predicted_velocity_reduction = (
            drift["predicted_velocity_reduction"]
        )

        observation.drift_confidence = (
            drift["confidence"]
        )

        # ====================================================
        # STEP 2
        # PRELIMINARY MAP ERROR
        # ====================================================

        if candidates:

            map_error = min(
                c.distance
                for c in candidates
            )

        else:

            map_error = 1000.0

        # ====================================================
        # STEP 3
        # CREATE RL STATE
        # ====================================================

        state = self.rl.discretize_state(
            observation,
            map_error
        )

        # ====================================================
        # STEP 4
        # RL ACTION
        # ====================================================

        action = self.rl.choose_action(
            state
        )

        self.rl.apply_action(
            action,
            self.hmm
        )

        # ====================================================
        # STEP 5
        # HMM INFERENCE
        # ====================================================

        result = self.hmm.infer(
            observation,
            candidates
        )

        if result is None:
            return None

        best_road = result["road"]

        probabilities = result[
            "probabilities"
        ]

        # ====================================================
        # STEP 6
        # MAP-MATCHED POSITION
        # ====================================================

        matched_lat = (
            best_road.latitude
        )

        matched_lon = (
            best_road.longitude
        )

        # ====================================================
        # STEP 7
        # CALCULATE ERROR
        # ====================================================

        current_error = (
            best_road.distance
        )

        # ====================================================
        # STEP 8
        # REWARD
        # ====================================================

        reward = self.calculate_reward(
            observation,
            current_error,
            best_road
        )

        # ====================================================
        # STEP 9
        # UPDATE RL
        # ====================================================

        next_state = self.rl.discretize_state(
            observation,
            current_error
        )

        if self.previous_state is not None:

            self.rl.update(
                self.previous_state,
                self.previous_action,
                reward,
                next_state
            )

        self.previous_state = state
        self.previous_action = action

        self.previous_error = current_error

        # ====================================================
        # OUTPUT
        # ====================================================

        return {
            "latitude": matched_lat,
            "longitude": matched_lon,

            "road_id": best_road.road_id,

            "road_probability": float(
                np.max(probabilities)
            ),

            "map_error": current_error,

            "predicted_drift":
                observation.predicted_drift,

            "velocity_reduction":
                observation.predicted_velocity_reduction,

            "drift_confidence":
                observation.drift_confidence,

            "rl_action":
                self.rl.actions[action],

            "reward": reward,

            "hmm_probabilities":
                probabilities.tolist()
        }

    # --------------------------------------------------------
    # REWARD FUNCTION
    # --------------------------------------------------------

    def calculate_reward(
        self,
        observation,
        map_error,
        road
    ):

        # Position consistency
        position_reward = -map_error

        # Heading consistency
        heading_error = angular_difference(
            observation.heading,
            road.road_heading
        )

        heading_reward = -(
            heading_error / 10.0
        )

        # Speed consistency
        speed_error = abs(
            observation.speed -
            road.road_speed
        )

        speed_reward = -(
            speed_error
        )

        # Penalize excessive drift
        drift_penalty = (
            observation.predicted_drift
            * 0.1
        )

        reward = (
            position_reward
            +
            heading_reward
            +
            speed_reward
            -
            drift_penalty
        )

        return reward


# ============================================================
# 7. SIMULATED ROAD NETWORK
# ============================================================

def create_simulated_roads():

    roads = [

        RoadCandidate(
            road_id="ROAD_A",
            latitude=13.0827,
            longitude=80.2707,
            road_heading=0.0,
            road_speed=12.0,
            distance=4.0
        ),

        RoadCandidate(
            road_id="ROAD_B",
            latitude=13.0828,
            longitude=80.2708,
            road_heading=90.0,
            road_speed=8.0,
            distance=15.0
        ),

        RoadCandidate(
            road_id="ROAD_C",
            latitude=13.0826,
            longitude=80.2706,
            road_heading=180.0,
            road_speed=10.0,
            distance=30.0
        ),

        RoadCandidate(
            road_id="ROAD_D",
            latitude=13.0829,
            longitude=80.2709,
            road_heading=270.0,
            road_speed=6.0,
            distance=50.0
        )
    ]

    return roads


# ============================================================
# 8. SIMULATION
# ============================================================

def run_simulation():

    system = HMMRLNavigationSystem()

    roads = create_simulated_roads()

    for t in range(100):

        # Simulated navigation measurements

        speed = (
            12.0
            +
            np.random.normal(0, 1)
        )

        acceleration = (
            np.random.normal(0, 0.5)
        )

        yaw_rate = (
            np.random.normal(0, 0.1)
        )

        heading = (
            np.random.normal(0, 5)
        )

        # Simulated GNSS degradation
        if 40 < t < 70:

            gnss_accuracy = 30.0

        else:

            gnss_accuracy = 5.0

        observation = NavigationObservation(

            timestamp=t * 0.1,

            latitude=13.0827
            + np.random.normal(0, 0.00005),

            longitude=80.2707
            + np.random.normal(0, 0.00005),

            speed=speed,

            heading=heading,

            acceleration=acceleration,

            yaw_rate=yaw_rate,

            gnss_accuracy=gnss_accuracy,

            gnss_confidence=
                1.0 / gnss_accuracy
        )

        # Artificially modify road distances
        #
        # During GNSS degradation,
        # drift grows.

        if 40 < t < 70:

            for road in roads:

                road.distance += (
                    np.random.uniform(0, 2)
                )

        result = system.process(
            observation,
            roads,
            dt=0.1
        )

        if result:

            print(
                f"\nTime: {t * 0.1:.1f}s"
            )

            print(
                f"Road: "
                f"{result['road_id']}"
            )

            print(
                f"Map Error: "
                f"{result['map_error']:.2f} m"
            )

            print(
                f"Predicted Drift: "
                f"{result['predicted_drift']:.3f} m"
            )

            print(
                f"Velocity Reduction: "
                f"{result['velocity_reduction']:.3f} m/s"
            )

            print(
                f"RL Action: "
                f"{result['rl_action']}"
            )

            print(
                f"Reward: "
                f"{result['reward']:.3f}"
            )


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":

    run_simulation()