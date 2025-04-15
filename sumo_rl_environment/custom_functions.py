"""
File that contains the custom Sumo environment.
This to ensure that the different training scripts can use the same environment.
"""

from typing import List
from collections import deque
import os
from typing import Callable, Optional, Tuple, Union
import numpy as np
import traci
import sumolib

from gymnasium import spaces
from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction, DefaultObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal

from sumo_rl_environment.custom_traffic_light import CustomTrafficSignal

class CustomObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)
        # self.previous_queue_length = []
        self.phase_id_buffer = None
        self.queue_history_len = 10
        self.previous_queue_lengths = None
        
    
    def custom_get_lanes_queue(self) -> List[float]:
        """Returns the queue [0,100] of the vehicles in the incoming lanes of the intersection.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        ts = self.ts
        lanes_queue = [
            ts.sumo.lane.getLastStepHaltingNumber(lane)
            / (ts.lanes_length[lane] / (ts.MIN_GAP + ts.sumo.lane.getLastStepLength(lane)))
            for lane in ts.lanes
        ]
        return [min(100, queue * 100) for queue in lanes_queue]

    def waiting_time_per_lane(self):
        """Returns the waiting time of all vehicles per lane
        
        Returns as list of lists
        """
        ts = self.ts
        wait_times_per_lane = []
        for lane in ts.lanes:
            veh_list = ts.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = []
            for veh in veh_list:
                veh_lane = ts.sumo.vehicle.getLaneID(veh)
                acc = ts.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in ts.env.vehicles:
                    ts.env.vehicles[veh] = {veh_lane: acc}
                else:
                    ts.env.vehicles[veh][veh_lane] = acc - sum(
                        [ts.env.vehicles[veh][lane] for lane in ts.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time.append(ts.env.vehicles[veh][veh_lane])
            wait_times_per_lane.append(wait_time)
        return wait_times_per_lane
    
    
    def get_observations_dict(self) -> dict:
        """Generates the observation as dictionary
        """
        if self.phase_id_buffer is None:
            # num_green_phases is only instantiated later
            self.phase_id_buffer = {i: deque([0] * 4, maxlen=4) for i in range(self.ts.num_green_phases)}
            self.previous_queue_lengths = {i: deque([0] * self.queue_history_len, maxlen=self.queue_history_len)
                                           for i in range(len(self.ts.lanes))}

        current_time = [self.ts.sumo.simulation.getTime()]
        
        current_phase_ids = []
        for phase in range(self.ts.num_green_phases):
            if self.ts.green_phase == phase:
                current_phase_ids.append(1)
                self.phase_id_buffer[phase].appendleft(1)
            else:
                current_phase_ids.append(0)
                self.phase_id_buffer[phase].appendleft(0)
        hist_phase_ids = []
        for phase in range(self.ts.num_green_phases):
            hist_phase_ids.extend(list(self.phase_id_buffer[phase]))
        # phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        # queue = self.ts.get_lanes_queue()
        queue = self.custom_get_lanes_queue()
        # if not self.previous_queue_length:
        #     self.previous_queue_length = [0] * len(queue)
        # delta_queue = [queue[i]- self.previous_queue_length[i] for i in range(len(queue))]  
        # self.previous_queue_length = queue
        queue_der = []  # TODO rename to derivative of queue
        for i, l in enumerate(queue):
            self.previous_queue_lengths[i].append(l)
            queue_diff = np.diff(self.previous_queue_lengths[i])
            # delta_queue.append(np.mean(queue_diff)) # THis seems to have collapsed the oc into only doing 1 action
            queue_der.extend(queue_diff)
        waiting_times = self.waiting_time_per_lane()
        
        return {
            # "current_time": current_time,
            "current_phase_ids": current_phase_ids,
            "hist_phase_ids": hist_phase_ids,
            # "phase_id": phase_id,
            "min_green": min_green,
            "density": density,
            "queue": queue,
            "queue_der": queue_der,
            "waiting_times": waiting_times,
            # "avg_waiting_time": [np.mean(waiting_time) for waiting_time in waiting_times],
            "average_speed": self.ts.get_average_speed()
        }

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        # TODO: create function that returns the observation as dict
        observation_dict = self.get_observations_dict()
        # current_time = observation_dict["current_time"]
        phase_ids = observation_dict["current_phase_ids"]
        # phase_ids = observation_dict["hist_phase_ids"]
        # min_green = observation_dict["min_green"]
        # density = observation_dict["density"]
        queue = observation_dict["queue"]
        # queue_der = observation_dict["queue_der"]
        # average_speed = observation_dict["average_speed"]
        waiting_times = []
        for waiting_time in observation_dict["waiting_times"]:
            if waiting_time:
                waiting_times.append(np.mean(waiting_time))
            else:
                waiting_times.append(0)
        # observation = np.array(phase_ids + queue + queue_der + waiting_times + [average_speed], dtype=np.float32)
        observation = np.array(phase_ids + queue + waiting_times, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros((self.ts.num_green_phases) + (2 * len(self.ts.lanes)) , dtype=np.float32),
            high=np.ones((self.ts.num_green_phases) + (2 * len(self.ts.lanes)), dtype=np.float32),
        )

def custom_reward_function(traffic_signal: CustomTrafficSignal):
    """Custom reward function that uses the pressure reward as a benchmark
    but penalizes based on the max waiting time of the lane 
    and the frequency in which the lights have changed
    """
    return traffic_signal.custom_reward()


def queue_based_reward_function(traffic_signal: CustomTrafficSignal):
    """Custom reward function that uses the pressure reward as a benchmark
    but penalizes based on the max waiting time of the lane 
    and the frequency in which the lights have changed
    """
    return traffic_signal.queue_based_reward()


def queue_based_reward2_function(traffic_signal: CustomTrafficSignal):
    """Custom reward function that uses the pressure reward as a benchmark
    but penalizes based on the max waiting time of the lane 
    and the frequency in which the lights have changed
    """
    return traffic_signal.queue_based_reward2()



def queue_based_reward3_function(traffic_signal: CustomTrafficSignal):
    """Custom reward function that uses the pressure reward as a benchmark
    but penalizes based on the max waiting time of the lane 
    and the frequency in which the lights have changed
    """
    return traffic_signal.queue_based_reward3()


def intelli_light_reward(traffic_signal: CustomTrafficSignal):
    return traffic_signal.intelli_light_reward()


def intelli_light_reward_prioritized(traffic_signal: CustomTrafficSignal):
    return traffic_signal.intelli_light_reward_prioritized()