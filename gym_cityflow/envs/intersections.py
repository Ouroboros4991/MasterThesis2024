"""
File containing the logic for intersections
"""

# Based on https://github.com/ramos-ai/explainable-drl-traffic-lights/blob/master/world.py
import cityflow


class Intersection:

    def __init__(
        self,
        eng: cityflow.Engine,
        id: str,
        intersection_dict: dict,
        interval: int,
        yellow_phase_duration: int,
    ):
        self.id = id
        self.interval = interval
        self.current_phase = 0
        self.target_phase = 0
        self.yellow_phase_duration = yellow_phase_duration
        self.steps_in_yellow = 0
        incoming_lanes, outgoing_lanes, tl_configs = self._extract_dict_data(
            intersection_dict
        )
        self.incoming_lanes = incoming_lanes
        self.outgoing_lanes = outgoing_lanes
        self.tl_configs = tl_configs
        self.eng = eng

    def set_phase(self, phase: int):
        """
        Update the phase of the intersection, taking into account the yellow phase
        """
        # We want to switch phase
        if phase != self.target_phase:
            self.target_phase = phase
        if self.target_phase != self.current_phase:
            if (
                self.steps_in_yellow >= self.yellow_phase_duration
            ):  # Yellow phase is done, switch to target phase
                self.current_phase = self.target_phase
                self.steps_in_yellow = 0
                self.eng.set_tl_phase(self.id, self.current_phase)
            elif (
                self.current_phase != self.yellow_phase
            ):  # Before switching, we need to go through the yellow phase
                self.current_phase = self.yellow_phase
                self.eng.set_tl_phase(self.id, self.current_phase)
            else:  # current phase is the yellow phase
                self.steps_in_yellow += self.interval

    def _extract_dict_data(self, intersection_dict: dict):
        incoming_lanes = set()
        outgoing_lanes = set()
        tl_configs = set()
        for road in intersection_dict["roadLinks"]:
            for lane in road["laneLinks"]:
                incoming_lanes.add(
                    road["startRoad"] + "_" + str(lane["startLaneIndex"])
                )
                outgoing_lanes.add(road["endRoad"] + "_" + str(lane["endLaneIndex"]))
        for index, tf in enumerate(intersection_dict["trafficLight"]["lightphases"]):
            if not tf["availableRoadLinks"]:
                tf_id = "do_nothing"
                self.yellow_phase = index
                tl_configs.add((index, tf_id))
            else:
                linked_types = []
                for road_link in tf["availableRoadLinks"]:
                    # Retrieve the info of the road that the traffic light is controlling
                    target_road = intersection_dict["roadLinks"][road_link]
                    linked_type = target_road["startRoad"] + "-" + target_road["type"]
                    linked_types.append(linked_type)
                tf_id = ";".join(linked_types)
                tl_configs.add((index, tf_id))
        return incoming_lanes, outgoing_lanes, tl_configs
