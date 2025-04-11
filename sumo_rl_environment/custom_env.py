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
        intelli_light_weight: dict = {}
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
        self.intelli_light_weight = intelli_light_weight

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
                    intelli_light_weight=self.intelli_light_weight,
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
                    intelli_light_weight=self.intelli_light_weight,
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
    
    
    
    def __init__(self, net_file, route_file, begin_time, num_seconds, use_gui=False, out_csv_name=None,
                 reward_fn: str = "intelli_light_reward",
                 intelli_light_weight: dict = {}):
        reward_mapping = {
            "pressure": "pressure",
            "intelli_light_reward" : intelli_light_reward,
            "intelli_light_reward_prioritized": intelli_light_reward_prioritized,
        }
        if not reward_fn:
            reward_fn = "intelli_light_reward"

        self.super_init(
            net_file=net_file,
            route_file=route_file,
            begin_time=begin_time,
            num_seconds=num_seconds,
            single_agent=False,
            add_per_agent_info=True,
            add_system_info=True,
            observation_class=CustomObservationFunction,
            use_gui=use_gui,
            additional_sumo_cmd='--tripinfo-output',
            # out_csv_name=out_csv_name,
            # reward_fn=custom_reward_function,
            # reward_fn=queue_based_reward_function,
            # reward_fn=queue_based_reward2_function,
            # reward_fn=queue_based_reward3_function,
            reward_fn=reward_mapping[reward_fn],
            intelli_light_weight=intelli_light_weight,
        )
        self.num_seconds = num_seconds
    
    
    def step(self, action: Union[dict, int]):
        """Apply the action(s) and then step the simulation for delta_time seconds.

        Args:
            action (Union[dict, int]): action(s) to be applied to the environment.
            If single_agent is True, action is an int, otherwise it expects a dict with keys corresponding to traffic signal ids.
        """
        observations, rewards, dones, info = super().step(action)
        terminated = dones["__all__"]
        truncated = terminated
        reward = np.mean(list(rewards.values()))
        return observations, reward, terminated, truncated, info
    
    @property
    def action_space(self):
        """Return the action space of a traffic signal.

        Only used in case of single-agent environment.
        """
        if not self.single_agent:
            return spaces.Dict({ts: self.traffic_signals[ts].action_space for ts in self.ts_ids})
        return self.traffic_signals[self.ts_ids[0]].action_space
    
    
    @property
    def observation_space(self):
        """Return the observation space of a traffic signal.

        Only used in case of single-agent environment.
        """
        if not self.single_agent:
            return spaces.Dict({ts: self.traffic_signals[ts].observation_space for ts in self.ts_ids})
        return self.traffic_signals[self.ts_ids[0]].observation_space
    
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
                    intelli_light_weight=self.intelli_light_weight,
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
                    intelli_light_weight=self.intelli_light_weight,
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
            return self._compute_observations(), self._compute_info()
