# Based on code from https://github.com/ramos-ai/explainable-drl-traffic-lights/blob/master/agent/fixedtime_agent.py

import math


class FixedTimeAgent:
    def __init__(self, n_actions, rotation_period, env, *args, **kwargs):
        self.n_actions = n_actions
        self.rotation_period = rotation_period
        self.counter = 0
        self.action = 1  # We skip the 0th action as it is used for the yellow phase
        intersections = env.intersections
        if len(intersections) != 1:
            raise ValueError(
                "FixedTimeAgent only supports single intersection scenarios"
            )
        self.intersection = list(intersections.values())[0]

    def predict(self, observation):
        """Predict the action for the current state.
        Args:
            observation (np.ndarray): The current state of the environment.
        Returns:
            int: The action to take.
        """
        if self.intersection.current_phase == self.action:
            self.counter += 1
        if self.counter >= self.rotation_period:
            self.action = (self.action + 1) % self.n_actions
            if self.action == 0:
                self.action = 1
            self.counter = 0
        return self.action, None
