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
        self.intersections = {}
        for i in range(len(self.roadnetDict["intersections"])):
            # check if intersection is controllable
            if self.roadnetDict["intersections"][i]["virtual"] == False:
                # for each roadLink in intersection store incoming lanes, outgoing lanes and direction in lists
                incomingLanes = []
                outgoingLanes = []
                directions = []
                for j in range(len(self.roadnetDict["intersections"][i]["roadLinks"])):
                    incomingRoads = []
                    outgoingRoads = []
                    directions.append(
                        self.roadnetDict["intersections"][i]["roadLinks"][j][
                            "direction"
                        ]
                    )
                    for k in range(
                        len(
                            self.roadnetDict["intersections"][i]["roadLinks"][j][
                                "laneLinks"
                            ]
                        )
                    ):
                        incomingRoads.append(
                            self.roadnetDict["intersections"][i]["roadLinks"][j][
                                "startRoad"
                            ]
                            + "_"
                            + str(
                                self.roadnetDict["intersections"][i]["roadLinks"][j][
                                    "laneLinks"
                                ][k]["startLaneIndex"]
                            )
                        )
                        outgoingRoads.append(
                            self.roadnetDict["intersections"][i]["roadLinks"][j][
                                "endRoad"
                            ]
                            + "_"
                            + str(
                                self.roadnetDict["intersections"][i]["roadLinks"][j][
                                    "laneLinks"
                                ][k]["endLaneIndex"]
                            )
                        )
                    incomingLanes.append(incomingRoads)
                    outgoingLanes.append(outgoingRoads)

                # add intersection to dict where key = intersection_id
                # value = no of lightPhases, incoming lane names, outgoing lane names, directions for each lane group
                self.intersections[self.roadnetDict["intersections"][i]["id"]] = [
                    [
                        len(
                            self.roadnetDict["intersections"][i]["trafficLight"][
                                "lightphases"
                            ]
                        )
                    ],
                    incomingLanes,
                    outgoingLanes,
                    directions,
                ]
        # setup intersectionNames list for agent actions
        self.intersectionNames = []
        for key in self.intersections:
            self.intersectionNames.append(key)

        # define action space MultiDiscrete()
        action_space_arr = []
        for key in self.intersections:
            action_space_arr.append(self.intersections[key][0][0])
        self.action_space = spaces.MultiDiscrete(action_space_arr)
        # define observation space
        # obs = self._get_observation().tolist()
        # self.observation_space = spaces.MultiDiscrete([1 for _ in obs])
        
        obs = self._get_observation()
        observationSpaceDict = {}
        for key, value in obs.items():
            observationSpaceDict[key] = spaces.MultiDiscrete([1 for _ in value])
        self.observation_space = spaces.Dict(observationSpaceDict)


    def step(self, action):
        # Check that input action size is equal to number of intersections
        if len(action) != len(self.intersectionNames):
            raise Warning("Action length not equal to number of intersections")

        # Set each trafficlight phase to specified action
        for i in range(len(self.intersectionNames)):
            self.eng.set_tl_phase(self.intersectionNames[i], action[i])

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
        observation = []
        obs_dict = {}
        for key in self.intersections:
            waitingIntersection = []
            for i in range(len(self.intersections[key][1])):
                for j in range(len(self.intersections[key][1][i])):
                    waitingIntersection.extend(
                        [
                            lane_waiting_vehicles_dict[
                                self.intersections[key][1][i][j]
                            ],
                            lane_waiting_vehicles_dict[
                                self.intersections[key][2][i][j]
                            ],
                        ]
                    )
            obs_dict[key] = np.array(waitingIntersection)
            observation.extend(waitingIntersection)

        # return np.array(observation)
        return obs_dict

    def get_reward(self):
        # We use pressure as reward
        # Pressure is defined as the difference between the number of cars entering the intersection 
        # and the number of cars leaving the intersection
        vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        incoming_cars = 0
        outgoing_cars = 0
        for _, values in self.intersections.items():
            # Lightphases, incoming lanes, outgoing lanes, directions
            _, incoming, outgoing, _ = values
            for lane in incoming:
                for road in lane:
                    incoming_cars += vehicle_count[road]
            for lane in outgoing:
                for road in lane:
                    outgoing_cars += vehicle_count[road]
        pressure = outgoing_cars - incoming_cars  # we want a possitive reward if more cars are leaving then arriving
        return pressure

    def seed(self, seed=None):
        self.eng.set_random_seed(seed)
