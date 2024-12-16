"""This file contains a custom sumo RL for the option training"""

import os
import logging
from typing import Callable, Optional, Tuple, Union

import numpy as np
import sumolib
import traci
from gymnasium import spaces
from gymnasium.utils import EzPickle
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from supersuit import pad_action_space_v0
from sumo_rl.environment.env import SumoEnvironmentPZ
from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal

from sumo_rl import SumoEnvironment

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


class CompleteObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, env: SumoEnvironment):
        """Initialize default observation function."""
        self.env = env

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        all_phase_id = []
        all_min_green = []
        all_density = []
        all_queue = []

        for ts in self.env.traffic_signals.values():
            phase_id = [
                1 if ts.green_phase == i else 0 for i in range(ts.num_green_phases)
            ]  # one-hot encoding
            min_green = [
                (
                    0
                    if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time
                    else 1
                )
            ]
            density = ts.get_lanes_density()
            queue = ts.get_lanes_queue()

            all_phase_id += phase_id
            all_min_green += min_green
            all_density += density
            all_queue += queue
        observation = np.array(
            all_phase_id + all_min_green + all_density + all_queue, dtype=np.float32
        )
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        size = 0
        for ts in self.env.traffic_signals.values():
            size += ts.num_green_phases
            size += 1
            size += 2 * len(ts.lanes)
        return spaces.Box(
            low=np.zeros(size, dtype=np.float32),
            high=np.ones(size, dtype=np.float32),
        )


class AllKnowningTrafficSignal(TrafficSignal):

    def __init__(self, *args, **kwargs):
        super(AllKnowningTrafficSignal, self).__init__(*args, **kwargs)
        self.observation_fn = CompleteObservationFunction(self.env)
        self.observation_space = self.observation_fn.observation_space()

    def _observation_fn_default(self):
        return self.observation_fn()


class CustomSumoEnvironment(SumoEnvironment):

    def __init__(self, *args, **kwargs):
        super(CustomSumoEnvironment, self).__init__(*args, **kwargs)
        if LIBSUMO:
            traci.start(
                [sumolib.checkBinary("sumo"), "-n", self._net]
            )  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start(
                [sumolib.checkBinary("sumo"), "-n", self._net],
                label="init_connection" + self.label,
            )
            conn = traci.getConnection("init_connection" + self.label)
        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {
                ts: AllKnowningTrafficSignal(
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
                ts: AllKnowningTrafficSignal(
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

    # Remove the dependency of the flag single_agent to apply the action
    def _apply_actions(self, actions):
        """Set the next green phase for the traffic signals.

        Args:
            actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                     If multiagent, actions is a dict {ts_id : greenPhase}
        """
        if isinstance(actions, int):
            if self.traffic_signals[self.ts_ids[0]].time_to_act:
                self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        elif isinstance(actions, dict):
            for ts, action in actions.items():
                if self.traffic_signals[ts].time_to_act:
                    self.traffic_signals[ts].set_next_phase(action)
        else:
            raise Exception("Invalid action type provided")

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
                ts: AllKnowningTrafficSignal(
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
                ts: AllKnowningTrafficSignal(
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

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]], self._compute_info()
        else:
            return self._compute_observations()

    def _compute_observations(self):
        self.observations.update(
            {
                ts: self.traffic_signals[ts].compute_observation()
                for ts in self.ts_ids
                if self.traffic_signals[ts].time_to_act or self.fixed_ts
            }
        )
        return {
            ts: self.observations[ts].copy()
            for ts in self.observations.keys()
            if self.traffic_signals[ts].time_to_act or self.fixed_ts
        }


class CustomSumoEnvironmentPZ(SumoEnvironmentPZ):

    def __init__(self, **kwargs):
        """Initialize the environment."""
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()
        self.env = CustomSumoEnvironment(**self._kwargs)

        self.agents = self.env.ts_ids
        self.possible_agents = self.env.ts_ids
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {
            a: self.env.observation_spaces(a) for a in self.agents
        }
        print("TESTING", self.observation_spaces)
        # dicts
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}


def custom_pz_env(**kwargs):
    """Instantiate a PettingoZoo environment."""
    env = CustomSumoEnvironmentPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    env = pad_action_space_v0(env)
    return env


custom_parallel_env = parallel_wrapper_fn(custom_pz_env)
