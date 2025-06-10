"""Class containing the logic needed for a MaxPressure.
Based on: https://github.com/docwza/sumolights
"""

import random


class MaxPressureAgent:

    def __init__(self, env, *args, **kwargs):
        self.env = env
        self.traffic_light_configs = {
            ts_id: {
                "current_phase": 0,
                "n_phases": traffic_light.num_green_phases,
                "phase_lanes": self.get_phase_lanes(traffic_light),
            }
            for ts_id, traffic_light in env.traffic_signals.items()
        }

    def get_phase_lanes(self, traffic_light) -> dict:
        """Get the incoming and outgoing lanes for each phase of the traffic light.

        Args:
            traffic_light (TrafficSignal): TrafficSignal object from sumo

        Raises:
            Exception: Failed to match lanes to phase state
            Exception: Failed to find green lanes for phase

        Returns:
            dict: Dict containig the incoming and outgoing lanes for each phase
        """
        ts_id = traffic_light.id
        # More information on these functions can be found here:
        # https://sumo.dlr.de/pydoc/traci._trafficlight.html#TrafficLightDomain-getControlledLinks
        controlled_lanes = traffic_light.sumo.trafficlight.getControlledLanes(ts_id)
        links = traffic_light.sumo.trafficlight.getControlledLinks(ts_id)
        phase_lanes = {}
        for phase_index, phase in enumerate(traffic_light.green_phases):
            incoming_lanes = set()
            if len(phase.state) != len(controlled_lanes):
                raise Exception("Failed to match lanes to phase state")
            for lane_state_index, lane_state_char in enumerate(str(phase.state)):
                if lane_state_char.lower() == "g":
                    lane = controlled_lanes[lane_state_index]
                    incoming_lanes.add(lane)
            # if not incoming_lanes:
            # raise Exception("No green lanes found for phase")
            outgoing_lanes = set()
            for lane in incoming_lanes:
                for link in links:
                    for incoming, outgoing, via in link:
                        if lane in incoming:
                            outgoing_lanes.add(outgoing)
            phase_lanes[phase_index] = {
                "incoming": incoming_lanes,
                "outgoing": outgoing_lanes,
            }
        return phase_lanes

    def get_pressure(self, inc_lanes, out_lanes) -> int:
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection."""
        return sum(
            self.env.sumo.lane.getLastStepVehicleNumber(lane) for lane in out_lanes
        ) - sum(self.env.sumo.lane.getLastStepVehicleNumber(lane) for lane in inc_lanes)

    def max_pressure(self, tf_id):
        """Returns the phase with the highest pressure.
        If all phases have 0 pressure, return a random phase.
        If multiple phases have the same pressure, select one randomly
        """
        phase_lanes = self.traffic_light_configs[tf_id]["phase_lanes"]
        n_phases = self.traffic_light_configs[tf_id]["n_phases"]
        phase_pressure = {}
        no_vehicle_phases = []
        # compute pressure for all green movements
        for phase_id in range(n_phases):
            inc_lanes = phase_lanes[phase_id]["incoming"]
            out_lanes = phase_lanes[phase_id]["outgoing"]
            pressure = abs(self.get_pressure(inc_lanes, out_lanes))
            phase_pressure[phase_id] = pressure
            if pressure == 0:
                no_vehicle_phases.append(phase_id)
        ###if no vehicles randomly select a phase
        if len(no_vehicle_phases) == n_phases:
            return random.randint(0, n_phases - 1)
        else:
            # choose phase with max pressure
            # if two phases have equivalent pressure
            # select one with more green movements
            # return max(phase_pressure, key=lambda p:phase_pressure[p])
            phase_pressure = [(p, phase_pressure[p]) for p in phase_pressure]
            phase_pressure = sorted(phase_pressure, key=lambda p: p[1], reverse=True)
            phase_pressure = [p for p in phase_pressure if p[1] == phase_pressure[0][1]]
            return random.choice(phase_pressure)[0]

    def predict(self, observation):
        """Predict the action for the current state.
        Args:
            observation (np.ndarray): The current state of the environment.
        Returns:
            int: The action to take.
        """
        action = {}
        for ts_id, ts_config in self.traffic_light_configs.items():
            n_phases = ts_config["n_phases"]
            currently_green = observation[ts_id][0:n_phases]
            # only update the phase if we're in a green phase
            # This to avoid skipping phases because of the yellow phase
            if any(currently_green):
                max_pressure_phase = self.max_pressure(ts_id)
                ts_config["current_phase"] = max_pressure_phase
            action[ts_id] = ts_config["current_phase"]
        return action, None
