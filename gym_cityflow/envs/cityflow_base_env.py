# Based on https://github.com/MaxVanDijck/gym-cityflow/blob/main/gym_cityflow/envs/cityflow_env.py

import json
import tempfile
import cityflow
import gymnasium as gym
import numpy as np
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding


class CityflowGym(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config_dict, episode_steps):
        # steps per episode
        self.max_cars = 100000  # max cars per lane, used to define observation space

        self.steps_per_episode = episode_steps
        self.is_done = False
        self.current_step = 0

        self.config_dict = config_dict
        # open cityflow roadnet file into dict
        self.roadnetDict = json.load(
            open(self.config_dict["dir"] + self.config_dict["roadnetFile"])
        )
        self.flowDict = json.load(
            open(self.config_dict["dir"] + self.config_dict["flowFile"])
        )

        # create cityflow engine
        tmp_config_file = tempfile.NamedTemporaryFile(mode="w+")
        json.dump(config_dict, tmp_config_file)
        tmp_config_file.flush()
        self.eng = cityflow.Engine(tmp_config_file.name, thread_num=1)

        # create dict of controllable intersections and number of light phases
        self.intersections = self._generate_intersections_dict(
            self.roadnetDict["intersections"]
        )

        # define action space MultiDiscrete()
        action_space_dict = {
            tl_id: spaces.Discrete(len(tl["tl_configs"]))
            for tl_id, tl in self.intersections.items()
        }
        self.action_space = spaces.Dict(action_space_dict)

        self.observation_space = self._generate_observation_space()

    def _generate_observation_space(self):
        """
        Generate the observation space for the environment
        """
        obs = self._get_observation()
        observationSpaceDict = {}
        for key, value in obs.items():
            observationSpaceDict[key] = spaces.MultiDiscrete(
                [self.max_cars for _ in value]
            )
        return spaces.Dict(observationSpaceDict)

    def _generate_intersections_dict(self, intersections):
        """Convert the roadnetdict into a dict of controllable intersections
        with their number of lightphases and lane groups
        """
        intersection_dict = {}
        for intersection in intersections:
            if intersection["virtual"]:
                continue
            incoming_lanes = set()
            outgoing_lanes = set()
            tl_configs = set()
            for road in intersection["roadLinks"]:
                for lane in road["laneLinks"]:
                    incoming_lanes.add(
                        road["startRoad"] + "_" + str(lane["startLaneIndex"])
                    )
                    outgoing_lanes.add(
                        road["endRoad"] + "_" + str(lane["endLaneIndex"])
                    )
            for index, tf in enumerate(intersection["trafficLight"]["lightphases"]):
                if not tf["availableRoadLinks"]:
                    tf_id = "do_nothing"
                    tl_configs.add((index, tf_id))
                else:
                    linked_types = []
                    for road_link in tf["availableRoadLinks"]:
                        # Retrieve the info of the road that the traffic light is controlling
                        target_road = intersection["roadLinks"][road_link]
                        linked_type = (
                            target_road["startRoad"] + "-" + target_road["type"]
                        )
                        linked_types.append(linked_type)
                    tf_id = ";".join(linked_types)
                    tl_configs.add((index, tf_id))
            intersection_dict[intersection["id"]] = {
                "incoming_lanes": list(incoming_lanes),
                "outgoing_lanes": list(outgoing_lanes),
                "tl_configs": list(tl_configs),
            }

        return intersection_dict

    def _set_tf_phases(self, action):
        """
        Set the traffic light phases according to the action
        """
        # Validate action
        # Check that input action size is equal to number of intersections
        if len(action) != len(self.intersections):
            raise Warning("Action length not equal to number of intersections")

        # Validate action type
        if isinstance(action, dict):
            pass
        elif isinstance(action, list) or isinstance(action, np.ndarray):
            if isinstance(action[0], dict):
                action = action[0]
            else:
                raise Exception("Action should be a dict")
        else:
            raise Exception(f"Unknown action type provided {type(action)}")

        # Set each trafficlight phase to specified action
        for intersection_id, phase in action.items():
            self.eng.set_tl_phase(intersection_id, phase)

    def step(self, action):
        self._set_tf_phases(action)

        # env step
        self.eng.next_step()
        # observation
        self.observation = self._get_observation()

        # reward
        self.reward = self.get_reward()
        # Detect if Simulation is finshed for done variable
        self.current_step += 1

        if self.current_step + 1 == self.steps_per_episode:
            self.is_done = True

        # return observation, reward, done, info
        return self.observation, self.reward, self.is_done, False, {}

    def reset(self, *args, **kwargs):
        self.eng.reset(seed=False)
        self.is_done = False
        self.current_step = 0
        return self._get_observation(), {}

    def render(self, mode="human"):
        print("Current time: " + self.cityflow.get_current_time())

    def _get_observation(self):
        # observation
        # get arrays of waiting cars on input lane vs waiting cars on output lane for each intersection
        lane_waiting_vehicles_dict = self.eng.get_lane_waiting_vehicle_count()
        obs_dict = {}
        for intersection_id, data in self.intersections.items():
            incoming_lanes = data["incoming_lanes"]
            intersection_obs = []
            for lane in incoming_lanes:
                if lane not in lane_waiting_vehicles_dict:
                    raise Exception(
                        f"lane {lane} not found in lane_waiting_vehicles_dict"
                    )
                waiting_cars = lane_waiting_vehicles_dict[lane]
                intersection_obs.append(waiting_cars)
            obs_dict[intersection_id] = np.array(intersection_obs)
            # TODO: add logic to include the change in the number of cars in the intersection
        return obs_dict

    def get_reward(self):
        # We use pressure as reward
        # Pressure is defined as the difference between the number of cars entering the intersection
        # and the number of cars leaving the intersection
        vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        incoming_cars = 0
        outgoing_cars = 0
        for _, data in self.intersections.items():
            for lane in data["incoming_lanes"]:
                incoming_cars += vehicle_count[lane]
            for lane in data["outgoing_lanes"]:
                outgoing_cars += vehicle_count[lane]
        pressure = (
            outgoing_cars - incoming_cars
        )  # we want a possitive reward if more cars are leaving then arriving
        return pressure

    def seed(self, seed=None):
        self.eng.set_random_seed(seed)
