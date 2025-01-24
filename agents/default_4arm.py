"""Class containing the logic a normal traffic lights follows.
Based on https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html
"""

class FourArmIntersection:
    
    def __init__(self, action_space, green_duration, *args, **kwargs):
        self.current_phase = 0
        self.number_phases = action_space.n
        self.green_duration = green_duration
        self.green_steps = 0
        
    
    
    def predict(self, observation):
        """Predict the action for the current state.
        Args:
            observation (np.ndarray): The current state of the environment.
        Returns:
            int: The action to take.
        """
        currently_green = observation[0:self.number_phases]
        # only update the phase if we're in a green phase
        # This to avoid skipping phases because of the yellow phase
        if any(currently_green):
            self.green_steps += 1
            if self.green_steps > self.green_duration:
                self.green_steps = 0
                self.current_phase = (self.current_phase + 1) % self.number_phases
        action = self.current_phase
        return action, None