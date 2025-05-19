"""Implementation of Self Organizing traffic lights"""


class SOTLPlatoonAgent:
    def __init__(self, env, green_duration: int = 3, *args, **kwargs):
        self.env = env
        self.traffic_light_configs = {
            ts_id: {
                "current_phase": 0,
                "n_phases": traffic_light.num_green_phases,
                "g_min": green_duration,
                "time_in_phase": 0,
                "theta": 9,  # threshold to change signal (veh*s)
                "omega": 1,  # sotl param (veh*s)
                "mu": 3,  # sotl param(veh*s)
                "kappa": 0,
                "phase_lanes": self.get_phase_lanes(traffic_light),
                "phase_red_lanes": self.get_phase_red_lanes(traffic_light),
            }
            for ts_id, traffic_light in env.traffic_signals.items()
        }

    def get_phase_red_lanes(self, traffic_light) -> dict:
        """Create a dict with the red lanes for each phase of the traffic light.

        Args:
            traffic_light (TrafficSignal): TrafficSignal object from sumo

        Raises:
            Exception: Failed to match lanes to phase state
            Exception: Failed to find green lanes for phase

        Returns:
            dict: Dict containing the red lanes for each phase
        """
        ts_id = traffic_light.id
        # More information on these functions can be found here:
        # https://sumo.dlr.de/pydoc/traci._trafficlight.html#TrafficLightDomain-getControlledLinks
        controlled_lanes = traffic_light.sumo.trafficlight.getControlledLanes(ts_id)
        phase_lanes = {}
        for phase_index, phase in enumerate(
            traffic_light.sumo.trafficlight.getAllProgramLogics(traffic_light.id)[
                0
            ].phases
        ):
            red_lanes = set()
            if len(phase.state) != len(controlled_lanes):
                raise Exception("Failed to match lanes to phase state")
            for lane_state_index, lane_state_char in enumerate(str(phase.state)):
                if lane_state_char.lower() == "r":
                    lane = controlled_lanes[lane_state_index]
                    red_lanes.add(lane)
            # if not red_lanes:
            #     print(phase.state)
            #     raise Exception("No red lanes found for phase")
            phase_lanes[phase_index] = red_lanes
        return phase_lanes

    def next_phase(self, tf_id: str) -> int:
        """Calculate the next phase of the given traffic light"""
        g_min = self.traffic_light_configs[tf_id]["g_min"]
        mu = self.traffic_light_configs[tf_id]["mu"]
        theta = self.traffic_light_configs[tf_id]["theta"]
        kappa = self.traffic_light_configs[tf_id]["kappa"]
        time_in_phase = self.traffic_light_configs[tf_id]["time_in_phase"]
        current_phase = self.traffic_light_configs[tf_id]["current_phase"]
        n_phases = self.traffic_light_configs[tf_id]["n_phases"]
        # stay in green phase for
        # minimum amount of time
        next_phase = current_phase
        if time_in_phase >= g_min:
            n = self.approaching_vehicles(tf_id)
            # if too many vehicles approaching green or no vehicles, go straight to kappa check
            if n > mu or n == 0:
                if kappa > theta:
                    next_phase = (current_phase + 1) % n_phases
                    self.traffic_light_configs[tf_id]["kappa"] = 0
                    self.traffic_light_configs[tf_id]["time_in_phase"] = 0
        return next_phase

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
            phase_red_lanes = ts_config["phase_red_lanes"]
            currently_green = observation[ts_id][0:n_phases]
            # only update the phase if we're in a green phase
            # This to avoid skipping phases because of the yellow phase
            if any(currently_green):
                ts_config["time_in_phase"] += 1
                ts_config["current_phase"] = self.next_phase(ts_id)
                ts_config["kappa"] += sum(
                    [
                        self.env.sumo.lane.getLastStepVehicleNumber(lane)
                        for lane in phase_red_lanes[ts_config["current_phase"]]
                    ]
                )
            action[ts_id] = ts_config["current_phase"]
        return action, None

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
            if not incoming_lanes:
                raise Exception("No green lanes found for phase")
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

    def approaching_vehicles(self, tf_id: str) -> int:
        """count the number of vehicles approaching
        the intersection in green lanes

        Args:
            tf_id (str): Target traffic light

        Returns:
            int: Number of approaching vehicles
        """
        phase_lanes = self.traffic_light_configs[tf_id]["phase_lanes"]
        current_phase = self.traffic_light_configs[tf_id]["current_phase"]

        inc_lanes = phase_lanes[current_phase]["incoming"]
        approaching_v = sum(
            [self.env.sumo.lane.getLastStepVehicleNumber(lane) for lane in inc_lanes]
        )
        return approaching_v
