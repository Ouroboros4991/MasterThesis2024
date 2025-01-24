"""
File that contains the custom Sumo environment.
This to ensure that the different training scripts can use the same environment.
"""

import numpy as np
from gymnasium import spaces
from sumo_rl.environment.traffic_signal import TrafficSignal
from sumo_rl.environment.observations import ObservationFunction

from sumo_rl import SumoEnvironment


class CustomObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)
        self.previous_queue_length = []

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        if not self.previous_queue_length:
            self.previous_queue_length = [0] * len(queue)
        delta_queue = [queue[i]- self.previous_queue_length[i] for i in range(len(queue))]        
        observation = np.array(phase_id + min_green + density + queue + delta_queue, dtype=np.float32)
        self.previous_queue_length = queue
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 3 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 3 * len(self.ts.lanes), dtype=np.float32),
        )

class CustomSumoEnvironment(SumoEnvironment):
    def __init__(self, net_file, route_file, begin_time, num_seconds, use_gui=False, out_csv_name=None):
        super().__init__(
            net_file=net_file,
            route_file=route_file,
            begin_time=begin_time,
            num_seconds=num_seconds,
            single_agent=True,
            add_per_agent_info=True,
            add_system_info=True,
            reward_fn='pressure',
            observation_class=CustomObservationFunction,
            use_gui=use_gui,
            additional_sumo_cmd='--tripinfo-output',
            # out_csv_name=out_csv_name
        )
        