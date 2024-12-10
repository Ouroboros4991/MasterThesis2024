"""This file contains a custom sumo RL for the option training"""

from sumo_rl import SumoEnvironment

class CustomSumoEnvironment(SumoEnvironment):
    
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
            raise Exception('Invalid action type provided')