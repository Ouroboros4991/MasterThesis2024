"""
This script generates the meta data for the traffic network. This includes the names of each lane and data on the actions.
This meta data can then be used to improve the visualizations.
"""
import argparse
import json
import pathlib

import xmltodict

from configs import ROUTE_SETTINGS
from utils import utils


def generate_lane_dicts(tf) -> dict:
    """Extract the lane info from the traffic signal object

    Args:
        tf (TrafficSignal): TrafficSignalObject

    Returns:
        dict: Dict containing the lane info.
    """    

    incoming_lane_dict = {}
    cleaned_incoming_lane_dict = {}
    cleaned_outgoing_lane_dict = {}
    outgoing_lane_dict = {}
    lane_index = 0
    for lane_index, lane_id in enumerate(tf.sumo.trafficlight.getControlledLanes(tf.id)):
        incoming_lane_dict[lane_index] = lane_id
    for lane_index, lane_id in enumerate(tf.out_lanes):
        outgoing_lane_dict[lane_index] = lane_id
    for lane_index, lane_id in enumerate(list(set(tf.sumo.trafficlight.getControlledLanes(tf.id)))):
        cleaned_incoming_lane_dict[lane_index] = lane_id
    for lane_index, lane_id in enumerate(list(set(tf.out_lanes))):
        cleaned_outgoing_lane_dict[lane_index] = lane_id
    return {
        "incoming": incoming_lane_dict,
        "cleaned_incoming": cleaned_incoming_lane_dict,
        "outgoing": outgoing_lane_dict,
        "cleaned_outgoing": cleaned_outgoing_lane_dict
    }


def generate_phase_data(tf, lanes: list) -> dict:
    """Get the phase data for the traffic signal

    Args:
        tf (TrafficSignal): TrafficSignalObject
        lanes (list): List of lanes

    Returns:
        dict: Phase data
    """    
    phase_data = {}
    for phase_id, phase in enumerate(tf.green_phases):
        lanes_green = set()
        for l, char in enumerate(phase.state):
            if char.lower() == 'g':
                lanes_green.add(lanes[l])
        
        phase_data[phase_id] = {
            "full_phase": phase.state,
            "lanes_green": list(lanes_green)
        }
    return phase_data

def generate_meta_data(traffic: str) -> dict:
    meta_data = {}
    env = utils.create_env(traffic)
    for tf_id, tf in env.traffic_signals.items():
        meta_data[tf_id] = {}
        lane_data = generate_lane_dicts(tf)
        meta_data[tf_id]["lanes"] = lane_data
        meta_data[tf_id]["phases"] = generate_phase_data(tf, lane_data["incoming"])
    json.dump(meta_data, open(f"meta/{traffic}.json", "w"), indent=4)

    


if __name__ == "__main__":
    pathlib.Path("./meta").mkdir(parents=True, exist_ok=True)

    possible_scenarios = list(ROUTE_SETTINGS.keys())
    possible_scenarios.append('all')
    parser = argparse.ArgumentParser(
                    description='Generate meta data jsons for a traffic network. This includes the names of each lane and data on the actions.',
                    )
    parser.add_argument('-t', '--traffic',
                        choices=possible_scenarios,
                        required=True) 
    args = parser.parse_args()

    if args.traffic == 'all':
        for scenario in ["cologne1", "cologne3", "cologne8",
                         "ingolstadt1", "ingolstadt7", "ingolstadt21",
                         "hangzhou_1x1_bc-tyc_18041607_1h", "jinan"]:
        # for scenario in ROUTE_SETTINGS.keys():
            generate_meta_data(scenario)
    else:
        generate_meta_data(args.traffic)
            