"""Class containing the logic a normal traffic lights follows.
Based on https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html
"""

class FourArmIntersection:
    
    def __init__(self, action_space, *args, **kwargs):
        self.current_phase = 0
        self.number_phases = action_space.n
    
    
    def predict(self, observation):
        """Predict the action for the current state.
        Args:
            observation (np.ndarray): The current state of the environment.
        Returns:
            int: The action to take.
        """
        action = self.current_phase
        self.current_phase = (self.current_phase + 1) % self.number_phases
        return action, None