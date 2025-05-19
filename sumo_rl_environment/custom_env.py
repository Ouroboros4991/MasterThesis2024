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
from sumo_rl_environment.custom_functions import (
    custom_reward_function,
    queue_based_reward_function,
    queue_based_reward2_function,
    queue_based_reward3_function,
    intelli_light_reward,
    intelli_light_reward_prioritized,
    intelli_light_prcol_reward,
    CustomObservationFunction,
    custom_waiting_time_reward
) 

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
            "custom_waiting_time": custom_waiting_time_reward,
            "intelli_light_prcol_reward": intelli_light_prcol_reward,
        }
        if not reward_fn:
            reward_fn = "intelli_light_reward"
        
        additional_sumo_cmd = [
            # "--device.rerouting.probability 1.0",
            # "--device.rerouting.period 60",
            # "--device.rerouting.adaptation-interval 10",
            # "--device.rerouting.adaptation-weight 0.2",
            "--tripinfo-output",
        ]
        # additional_sumo_cmd = "--tripinfo-output " \
        #       "--device.rerouting.probability 1.0 " \
        #       "--device.rerouting.period 60 " \
        #       "--device.rerouting.adaptation-interval 10 " \
        #       "--device.rerouting.adaptation-weight 0.2"

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
            additional_sumo_cmd=additional_sumo_cmd,
            # out_csv_name=out_csv_name,
            # reward_fn=custom_reward_function,
            # reward_fn=queue_based_reward_function,
            # reward_fn=queue_based_reward2_function,
            # reward_fn=queue_based_reward3_function,
            reward_fn=reward_mapping[reward_fn],
            intelli_light_weight=intelli_light_weight,
        )
        self.num_seconds = num_seconds
    
    def _start_simulation(self):
        sumo_cmd = [
            self._sumo_binary,
            "-n",
            self._net,
            "-r",
            self._route,
            "--max-depart-delay",
            str(self.max_depart_delay),
            "--waiting-time-memory",
            str(self.waiting_time_memory),
            "--time-to-teleport",
            str(self.time_to_teleport),
        ]
        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
        if self.additional_sumo_cmd is not None:
            if isinstance(self.additional_sumo_cmd, str):
                sumo_cmd.extend(self.additional_sumo_cmd.split())
            elif isinstance(self.additional_sumo_cmd, list):
                sumo_cmd.extend(self.additional_sumo_cmd)
            else:
                raise ValueError("additional_sumo_cmd must be a string or a list of strings.")
        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            if self.render_mode == "rgb_array":
                sumo_cmd.extend(["--window-size", f"{self.virtual_display[0]},{self.virtual_display[1]}"])
                from pyvirtualdisplay.smartdisplay import SmartDisplay

                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui or self.render_mode is not None:
            if "DEFAULT_VIEW" not in dir(traci.gui):  # traci.gui.DEFAULT_VIEW is not defined in libsumo
                traci.gui.DEFAULT_VIEW = "View #0"
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
    
    
    def step(self, action: Union[dict, int]):
        """Apply the action(s) and then step the simulation for delta_time seconds.

        Args:
            action (Union[dict, int]): action(s) to be applied to the environment.
            If single_agent is True, action is an int, otherwise it expects a dict with keys corresponding to traffic signal ids.
        """
        observations, rewards, dones, info = super().step(action)
        terminated = dones["__all__"]
        truncated = terminated
        # print(rewards)
        # reward = np.mean(list(rewards.values()))
        
        # Can't use fairness because the bad one will always perform worse?
        mean_reward = np.mean(list(rewards.values()))
        # Fairness enhanced reward
        # reward = 0
        # alpha = 0.5
        # for r in list(rewards.values()):
        #     reward += (r - alpha * (r - mean_reward)**2) 

        reward = mean_reward
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


class BrokenLightEnvironment(CustomSumoEnvironment):
    
    def __init__(self, broken_light_start: int=None, broken_light_end: int=None, *args, **kwargs):
        """Initialize the environment with a broken traffic light.

        Args:
            broken_light_start (int): time when the traffic light is broken.
            broken_light_end (int): time when the traffic light is fixed.
        """
        super().__init__(*args, **kwargs)
        self.broken_light_start = broken_light_start
        self.broken_light_end = broken_light_end
        self.broken_light_id = None            
        self.broken_light_period_specified = self.broken_light_start is not None and self.broken_light_end is not None
    
    def step(self, action: Union[dict, int]):
        """Apply the action(s) and then step the simulation for delta_time seconds.

        Args:
            action (Union[dict, int]): action(s) to be applied to the environment.
            If single_agent is True, action is an int, otherwise it expects a dict with keys corresponding to traffic signal ids.
        """
        should_replace = False
        if self.broken_light_period_specified:
            if self.sim_step >= self.broken_light_start and self.sim_step <= self.broken_light_end:
                should_replace = True
        else:
            should_replace = True
        if should_replace:
            if not self.broken_light_id:
                tf_ids = list(action.keys())
                middle = (len(tf_ids)-1) // 2
                self.broken_light_id = tf_ids[middle]
                print(f"Broken light id: {self.broken_light_id}")
            # Override the actions to keep 1 stop light on red as if the junction was blocked
            # TODO: update the logic so that this junction is only blocked for a certain amount of time
            action[self.broken_light_id ] = -1
        
        
        observations, reward, terminated, truncated, info = super().step(action)
        return observations, reward, terminated, truncated, info
    