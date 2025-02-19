"""
File that contains the custom Sumo environment.
This to ensure that the different training scripts can use the same environment.
"""

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


class CustomObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)
        self.previous_queue_length = []
        self.phase_id_buffer = None    
    
    def get_observations_dict(self) -> dict:
        """Generates the observation as dictionary
        """
        if self.phase_id_buffer is None:
            # num_green_phases is only instantiated later
            self.phase_id_buffer = {i: deque([0] * 4, maxlen=4) for i in range(self.ts.num_green_phases)}

        current_time = [self.ts.sumo.simulation.getTime()]
        
        for phase in range(self.ts.num_green_phases):
            if self.ts.green_phase == phase:
                self.phase_id_buffer[phase].appendleft(1)
            else:
                self.phase_id_buffer[phase].appendleft(0)
        phase_ids = []
        for phase in range(self.ts.num_green_phases):
            phase_ids.extend(list(self.phase_id_buffer[phase]))
        # phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        if not self.previous_queue_length:
            self.previous_queue_length = [0] * len(queue)
        delta_queue = [queue[i]- self.previous_queue_length[i] for i in range(len(queue))]  
        self.previous_queue_length = queue
        return {
            # "current_time": current_time,
            "phase_ids": phase_ids,
            # "phase_id": phase_id,
            "min_green": min_green,
            "density": density,
            "queue": queue,
            "delta_queue": delta_queue
        }

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        # TODO: create function that returns the observation as dict
        observation_dict = self.get_observations_dict()
        # current_time = observation_dict["current_time"]
        phase_ids = observation_dict["phase_ids"]
        # phase_id = observation_dict["phase_id"]
        min_green = observation_dict["min_green"]
        density = observation_dict["density"]
        queue = observation_dict["queue"]
        delta_queue = observation_dict["delta_queue"]
        observation = np.array(phase_ids + min_green + density + queue + delta_queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(4*self.ts.num_green_phases + 1 + 3 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(4*self.ts.num_green_phases + 1 + 3 * len(self.ts.lanes), dtype=np.float32),
        )

class CustomTrafficSignal(TrafficSignal):
    """Traffic signal class with some additional logic for the custom reward
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Keep track of how often a phase has been changed
        self.phases_changes = {i: [] for i in range(self.num_green_phases)}
        self.current_freq_phase = self.green_phase
        self.steps_in_current_phase = 0
        
        self.total_length = sum([self.sumo.lane.getLength(lane) for lane in self.lanes])
    
    def update(self):
        super().update()
        if self.current_freq_phase == self.green_phase:
            self.steps_in_current_phase += 1
        else:
            self.phases_changes[self.current_freq_phase].append(self.steps_in_current_phase)
            self.current_freq_phase = self.green_phase
            self.steps_in_current_phase = 0


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
        # reward -= 0.1 * scaled_max_waiting_time

        # # Decrease the reward further based on how often the phases have changed
        # # Penalize more frequent changes
        reward -= 5 * freq_penalty
        return reward


def custom_reward_function(traffic_signal: CustomTrafficSignal):
    """Custom reward function that uses the pressure reward as a benchmark
    but penalizes based on the max waiting time of the lane 
    and the frequency in which the lights have changed
    """
    return traffic_signal.custom_reward()


LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


class CustomSumoEnvironment(SumoEnvironment):
    
    def super_init(
        self,
        net_file: str,
        route_file: str,
        out_csv_name: Optional[str] = None,
        use_gui: bool = False,
        virtual_display: Tuple[int, int] = (3200, 1800),
        begin_time: int = 0,
        num_seconds: int = 20000,
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 50,
        single_agent: bool = False,
        reward_fn: Union[str, Callable, dict] = "diff-waiting-time",
        observation_class: ObservationFunction = DefaultObservationFunction,
        add_system_info: bool = True,
        add_per_agent_info: bool = True,
        sumo_seed: Union[str, int] = "random",
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        additional_sumo_cmd: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Original init function of the SumoEnvironment class"""
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Invalid render mode."
        self.render_mode = render_mode
        self.virtual_display = virtual_display
        self.disp = None

        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")

        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."

        self.begin_time = begin_time
        self.sim_max_time = begin_time + num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.waiting_time_memory = waiting_time_memory  # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.single_agent = single_agent
        self.reward_fn = reward_fn
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None

        if LIBSUMO:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net], label="init_connection" + self.label)
            conn = traci.getConnection("init_connection" + self.label)

        self.ts_ids = list(conn.trafficlight.getIDList())
        self.observation_class = observation_class

        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {
                ts: CustomTrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn[ts],
                    conn,
                )
                for ts in self.reward_fn.keys()
            }
        else:
            self.traffic_signals = {
                ts: CustomTrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn,
                    conn,
                )
                for ts in self.ts_ids
            }

        conn.close()

        self.vehicles = dict()
        self.reward_range = (-float("inf"), float("inf"))
        self.episode = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: None for ts in self.ts_ids}
    
    
    def __init__(self, net_file, route_file, begin_time, num_seconds, use_gui=False, out_csv_name=None):
        self.super_init(
            net_file=net_file,
            route_file=route_file,
            begin_time=begin_time,
            num_seconds=num_seconds,
            single_agent=False,
            add_per_agent_info=True,
            add_system_info=True,
            # reward_fn='pressure',
            observation_class=CustomObservationFunction,
            use_gui=use_gui,
            additional_sumo_cmd='--tripinfo-output',
            reward_fn=custom_reward_function,
            # out_csv_name=out_csv_name
        )
    
    def get_observations_dict(self):
        """Returns the observation dict per traffic signal."""
        return {
                ts: self.traffic_signals[ts].observation_fn.get_observations_dict()
                for ts in self.ts_ids
                if self.traffic_signals[ts].time_to_act or self.fixed_ts
            }

    
    def reset(self, seed: Optional[int] = None, **kwargs):
        """Reset the environment."""
        super().reset(seed=seed, **kwargs)

        if self.episode != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.episode)
        self.episode += 1
        self.metrics = []

        if seed is not None:
            self.sumo_seed = seed
        self._start_simulation()

        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {
                ts: CustomTrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn[ts],
                    self.sumo,
                )
                for ts in self.reward_fn.keys()
            }
        else:
            self.traffic_signals = {
                ts: CustomTrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn,
                    self.sumo,
                )
                for ts in self.ts_ids
            }

        self.vehicles = dict()
        self.num_arrived_vehicles = 0
        self.num_departed_vehicles = 0
        self.num_teleported_vehicles = 0

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]], self._compute_info()
        else:
            return self._compute_observations()
