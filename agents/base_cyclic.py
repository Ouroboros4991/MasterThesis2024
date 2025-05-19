"""Class containing the logic a normal traffic lights follows.
Based on https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html
"""

class CyclicAgent:
    
    def __init__(self, env, green_duration, *args, **kwargs):
        self.green_duration = green_duration
        print(env.traffic_signals)
        self.traffic_light_configs = {
            ts_id: {
                "time_in_phase": 0,
                "current_phase": 0,
                "number_phases": traffic_light.num_green_phases
            }
            for ts_id, traffic_light in env.traffic_signals.items()
        }
    
    def predict(self, observation):
        """Predict the action for the current state.
        Args:
            observation (np.ndarray): The current state of the environment.
        Returns:
            int: The action to take.
        """
        action = {}
        for ts_id, ts_config in self.traffic_light_configs.items():
            n_phases = ts_config["number_phases"]
            currently_green = []
            for i in range(n_phases):
                # current_phase = i * 4
                phase_to_check = i
                currently_green.append(observation[ts_id][phase_to_check])
            # currently_green = observation[ts_id][0:n_phases]
            # only update the phase if we're in a green phase
            # This to avoid skipping phases because of the yellow phase
            if any(currently_green):
                ts_config["time_in_phase"] += 1
                if ts_config["time_in_phase"] > self.green_duration:
                    ts_config["time_in_phase"] = 0
                    ts_config["current_phase"] = (ts_config["current_phase"] + 1) % n_phases
            action[ts_id] = ts_config["current_phase"]
        return action, None