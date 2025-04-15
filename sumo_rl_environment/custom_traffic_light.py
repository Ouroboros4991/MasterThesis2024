from typing import List
from collections import deque
import os
from typing import Callable, Optional, Tuple, Union
import numpy as np
import traci
import sumolib

from sumo_rl.environment.traffic_signal import TrafficSignal


class CustomTrafficSignal(TrafficSignal):
    """Traffic signal class with some additional logic for the custom reward
    """
    
    def __init__(self, *args, intelli_light_weight: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Keep track of how often a phase has been changed
        self.phases_changes = {i: [] for i in range(self.num_green_phases)}
        self.phase_changed = False
        self.current_freq_phase = self.green_phase
        self.steps_in_current_phase = 0
        
        self.total_length = sum([self.sumo.lane.getLength(lane) for lane in self.lanes])
        
        self.previous_queue_length = []
        
        self.intelli_light_weight = intelli_light_weight
        
        self.prev_car_incoming = {}
        
        self.red_state = "r" * len(self.green_phases[0].state)
    
    def update(self):
        super().update()
        if self.current_freq_phase == self.green_phase:
            self.steps_in_current_phase += 1
        else:
            self.phases_changes[self.current_freq_phase].append(self.steps_in_current_phase)
            self.current_freq_phase = self.green_phase
            self.steps_in_current_phase = 0
    
    
    def set_next_phase(self, new_phase: int):
        """Sets what will be the next green phase and sets yellow phase if the next phase is different than the current.

        Args:
            new_phase (int): Number between [0 ... num_green_phases]
        """
        new_phase = int(new_phase)
        if new_phase == -1:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.red_state)
            self.next_action_time = self.env.sim_step + self.delta_time
        elif self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            # self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
            )
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0


    def waiting_time_per_lane(self):
        """Returns the waiting time of all vehicles per lane
        
        Returns as list of lists
        """
        wait_times_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = []
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time.append(self.env.vehicles[veh][veh_lane])
            wait_times_per_lane.append(wait_time)
        return wait_times_per_lane
    
    
    def waiting_time_with_reset(self):
        """Returns the waiting time of all vehicles per lane
        
        Returns as list of lists
        """
        wait_times_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = []
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time.append(self.env.vehicles[veh][veh_lane])
            wait_times_per_lane.append(wait_time)
        return wait_times_per_lane

    def get_scaled_max_waiting_time(self, with_reset: bool = False):
        if with_reset:
            waiting_times = self.waiting_time_with_reset()
        else:
            waiting_times = self.waiting_time_per_lane()
        max_waiting_time_per_lane = []
        for lane_waiting_times in waiting_times:
            if lane_waiting_times:
                max_waiting_time_per_lane.append(max(lane_waiting_times))
            else:
                max_waiting_time_per_lane.append(0)
        max_waiting_time = max(max_waiting_time_per_lane)
        
        scaled_max_waiting_time = max_waiting_time / self.env.sim_max_time
        return scaled_max_waiting_time

    def get_scaled_waiting_time(self, with_reset: bool = False):
        if with_reset:
            waiting_times = self.waiting_time_with_reset()
        else:
            waiting_times = self.waiting_time_per_lane()
        max_waiting_time_per_lane = []
        for lane_waiting_times in waiting_times:
            if lane_waiting_times:
                max_waiting_time_per_lane.append(max(lane_waiting_times) / self.env.sim_max_time)
            else:
                max_waiting_time_per_lane.append(0)
        return max_waiting_time_per_lane
    
    def get_delay(self):
        delays = []
        for lane in self.lanes:
            delays.append(1 - self.sumo.lane.getLastStepMeanSpeed(lane) / self.sumo.lane.getMaxSpeed(lane))
        return delays
    
    
    def get_cars_leaving(self):
        """Returns the number of cars that have left the intersection since the last action"""
        cars_left = 0
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            prev_veh = self.prev_car_incoming.get(lane, [])
            for veh in veh_list:
                if veh not in prev_veh:
                    cars_left += 1
            self.prev_car_incoming[lane] = veh_list
        return cars_left
    
    def custom_reward(self):
        """Custom reward function that uses the pressure reward as a benchmark
        but penalizes based on the max waiting time of the lane 
        and the frequency in which the lights have changed
        """
        waiting_times = self.waiting_time_per_lane()
        max_waiting_time_per_lane = []
        for lane_waiting_times in waiting_times:
            if lane_waiting_times:
                max_waiting_time_per_lane.append(max(lane_waiting_times))
            else:
                max_waiting_time_per_lane.append(0)
        max_waiting_time = max(max_waiting_time_per_lane)
        scaled_max_waiting_time = max_waiting_time / self.env.sim_max_time
        
        avg_phase_changes = []
        for phase in range(self.num_green_phases):
            if self.phases_changes[phase]:
                avg_phase_changes.append(np.mean(self.phases_changes[phase]))
        if avg_phase_changes:
            min_avg_phase_change = min(avg_phase_changes)
            freq_penalty = 1 - (min_avg_phase_change / self.env.sim_max_time)
        else:
            freq_penalty = 0

        # Calculate the reward based on the pressure
        # Scale it with the total length of the cross road
        reward = 10 * (self.get_pressure() / self.total_length)
        # Decrease it by the max amount that a car has been waiting
        reward -= 5 * scaled_max_waiting_time

        # # Decrease the reward further based on how often the phases have changed
        # # Penalize more frequent changes
        # reward -= 5 * freq_penalty
        return reward   

    def queue_based_reward(self):      
        scaled_max_waiting_time = self.get_scaled_max_waiting_time()
        
        avg_phase_changes = []
        for phase in range(self.num_green_phases):
            if self.phases_changes[phase]:
                avg_phase_changes.append(np.mean(self.phases_changes[phase]))
        if avg_phase_changes:
            min_avg_phase_change = min(avg_phase_changes)
            freq_penalty = 1 - (min_avg_phase_change / self.env.sim_max_time)
        else:
            freq_penalty = 0

        # Calculate the reward based on the pressure
        # Scale it with the total length of the cross road
        queue_lengths = self.get_lanes_queue()
        # reward = 10 * -1 * max(queue_lengths)
        
        max_queue_lane = np.argmax(queue_lengths)
        base_reward = 0
        for i, l in enumerate(queue_lengths):
            queue_reward = 1 - l
            if i == max_queue_lane:
                queue_reward = queue_reward * 2
            base_reward += queue_reward
        base_reward = base_reward / len(queue_lengths)
        reward = 10 * base_reward

        
        # Decrease it by the max amount that a car has been waiting
        reward -= 5 * scaled_max_waiting_time

        # # Decrease the reward further based on how often the phases have changed
        # # Penalize more frequent changes
        # reward -= 5 * freq_penalty
        return reward

    
    def queue_based_reward2(self):      
        waiting_times = self.waiting_time_per_lane()
        max_waiting_time_per_lane = []
        for lane_waiting_times in waiting_times:
            if lane_waiting_times:
                max_waiting_time_per_lane.append(max(lane_waiting_times))
            else:
                max_waiting_time_per_lane.append(0)
        max_waiting_time = max(max_waiting_time_per_lane)
        scaled_max_waiting_time = max_waiting_time / self.env.sim_max_time

        avg_phase_changes = []
        for phase in range(self.num_green_phases):
            if self.phases_changes[phase]:
                avg_phase_changes.append(np.mean(self.phases_changes[phase]))
        if avg_phase_changes:
            min_avg_phase_change = min(avg_phase_changes)
            freq_penalty = 1 - (min_avg_phase_change / self.env.sim_max_time)
        else:
            freq_penalty = 0

        # Calculate the reward based on the pressure
        # Scale it with the total length of the cross road
        queue_lengths = self.get_lanes_queue()
        # reward = 10 * -1 * max(queue_lengths)
        
        max_queue_lane = np.argmax(queue_lengths)
        base_reward = 0
        if self.previous_queue_length:
            for i, l in enumerate(queue_lengths):
                queue_reward = self.previous_queue_length[i] - l
                if i == max_queue_lane:
                    queue_reward = queue_reward * 2
                base_reward += queue_reward
            base_reward = base_reward / len(queue_lengths)
        # TODO: check for a cleaner way
        self.previous_queue_length = queue_lengths
        reward = 10 * base_reward
        
        # Decrease it by the max amount that a car has been waiting
        reward -= 5 * scaled_max_waiting_time

        # # Decrease the reward further based on how often the phases have changed
        # # Penalize more frequent changes
        # reward -= 5 * freq_penalty
        return reward
    
    def queue_based_reward3(self):      
        scaled_max_waiting_time = self.get_scaled_max_waiting_time()
        # Calculate the reward based on the pressure
        # Scale it with the total length of the cross road
        queue_lengths = self.get_lanes_queue()
        # reward = 10 * -1 * max(queue_lengths)
        
        max_queue_lane = np.argmax(queue_lengths)
        base_reward = 0
        for i, l in enumerate(queue_lengths):
            queue_reward = 1 - l
            if i == max_queue_lane:
                queue_reward = queue_reward * 2
            base_reward += queue_reward
        base_reward = base_reward / len(queue_lengths)
        reward = 10 * base_reward

        
        # Decrease it by the max amount that a car has been waiting
        reward -= 5 * scaled_max_waiting_time
        # print("BASE_REWARD", base_reward, scaled_max_waiting_time)

        # # Decrease the reward further based on how often the phases have changed
        # # Penalize more frequent changes
        # reward -= 5 * freq_penalty
        return reward


    def intelli_light_reward(self):
        """IntelliLight reward function
        """
        waiting_times = self.get_scaled_waiting_time(with_reset=True)
        queue_lengths = self.get_lanes_queue()
        delay = self.get_delay()
        light_switches = int(self.steps_in_current_phase == 0)
        cars_leaving = self.get_cars_leaving()
        
        rewards = {
            # 'queue': np.mean(queue_lengths),
            'delay': np.mean(delay),
            'waiting_time': np.mean(waiting_times),
            'light_switches': light_switches,
            # 'cars_leaving': cars_leaving,
            # 'travel_time': np.sum(travel_durations),
        }
        reward = cars_leaving
        for key, value in rewards.items():
            reward_weight = self.intelli_light_weight.get(key, 1)
            # print(key, value, reward_weight)
            reward -= reward_weight * value
        return reward

    
    def intelli_light_reward_prioritized(self):
        """IntelliLight reward function
        """
        waiting_times = self.get_scaled_waiting_time(with_reset=True)
        queue_lengths = self.get_lanes_queue()
        delay = self.get_delay()
        light_switches = int(self.steps_in_current_phase == 0)
        cars_leaving = self.get_cars_leaving()
        
        max_queue_lane = np.argmax(queue_lengths)
        
        queue_lengths[max_queue_lane] = 2 * queue_lengths[max_queue_lane]
        delay[max_queue_lane] = 2 * delay[max_queue_lane]
        waiting_times[max_queue_lane] = 2 * waiting_times[max_queue_lane]
        
        rewards = {
            # 'queue': np.mean(queue_lengths),
            'delay': np.mean(delay),
            'waiting_time': np.mean(waiting_times),
            'light_switches': light_switches,
            # 'cars_leaving': cars_leaving,
            # 'travel_time': np.sum(travel_durations),
        }
        reward = cars_leaving
        for key, value in rewards.items():
            reward_weight = self.intelli_light_weight.get(key, 1)
            # print(key, value, reward_weight)
            reward -= reward_weight * value
        return reward