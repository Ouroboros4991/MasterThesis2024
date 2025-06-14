ROUTE_SETTINGS = {
    "custom-single-intersection": {
        "path": "sumo_rl_nets/custom-single-intersection/single-intersection.{type}.xml",
        "begin_time": 0,
        "end_time": 5000,
    },
    "custom-2way-single-intersection": {
        "path": "sumo_rl_nets/custom-2way-single-intersection/single-intersection.{type}.xml",
        "begin_time": 0,
        "end_time": 6000,
    },
    "custom-2way-single-intersection2": {
        "path": "sumo_rl_nets/custom-2way-single-intersection2/single-intersection.{type}.xml",
        "begin_time": 0,
        "end_time": 5000,
    },
    "custom-2way-single-intersection3": {
        "path": "sumo_rl_nets/custom-2way-single-intersection3/single-intersection.{type}.xml",
        "begin_time": 0,
        "end_time": 5000,
    },
    "custom-2way-single-intersection-low": {
        "path": "sumo_rl_nets/custom-2way-single-intersection-low/single-intersection.{type}.xml",
        "begin_time": 0,
        "end_time": 1000,
    },
    "custom-2way-single-intersection-high": {
        "path": "sumo_rl_nets/custom-2way-single-intersection-high/single-intersection.{type}.xml",
        "begin_time": 0,
        "end_time": 1000,
    },
    "custom-2way-single-intersection-low-emergency": {
        "path": "sumo_rl_nets/custom-2way-single-intersection-low-emergency/single-intersection.{type}.xml",
        "begin_time": 0,
        "end_time": 1000,
    },
    "custom-2way-single-intersection-high-emergency": {
        "path": "sumo_rl_nets/custom-2way-single-intersection-high-emergency/single-intersection.{type}.xml",
        "begin_time": 0,
        "end_time": 1000,
    },
    "single-intersection": {
        "path": "sumo_rl_nets/single-intersection/single-intersection.{type}.xml",
        "begin_time": 0,
        "end_time": 20000,
    },
    "cologne1": {
        "path": "sumo_rl_nets/RESCO/cologne1/cologne1.{type}.xml",
        "begin_time": 25200,
        "end_time": 28800,
    },
    "cologne3": {
        "path": "sumo_rl_nets/RESCO/cologne3/cologne3.{type}.xml",
        "begin_time": 25200,
        "end_time": 28800,
    },
    "cologne8": {
        "path": "sumo_rl_nets/RESCO/cologne8/cologne8.{type}.xml",
        "begin_time": 25200,
        "end_time": 28800,
    },
    "ingolstadt1": {
        "path": "sumo_rl_nets/RESCO/ingolstadt1/ingolstadt1.{type}.xml",
        "begin_time": 57600,
        "end_time": 61200,
    },
    "ingolstadt7": {
        "path": "sumo_rl_nets/RESCO/ingolstadt7/ingolstadt7.{type}.xml",
        "begin_time": 57600,
        "end_time": 61200,
    },
    "ingolstadt21": {
        "path": "sumo_rl_nets/RESCO/ingolstadt21/ingolstadt21.{type}.xml",
        "begin_time": 57600,
        "end_time": 61200,
    },
    "hangzhou_1x1_bc-tyc_18041607_1h": {
        "path": "sumo_rl_nets/hangzhou_1x1_bc-tyc_18041607_1h/hangzhou_1x1_bc-tyc_18041607_1h.{type}.xml",
        "begin_time": 0,
        "end_time": 3600,
    },
    "jinan": {
        "path": "sumo_rl_nets/jinan/jinan.{type}.xml",
        "begin_time": 0,
        "end_time": 3600,
    },
    "3x3grid": {
        "path": "sumo_rl_nets/3x3grid/3x3Grid2lanes.{type}.xml",
        "begin_time": 0,
        "end_time": 3600,  # TODO: increase to 26000
    },
    "3x3grid-specialized": {
        "path": "sumo_rl_nets/3x3grid-specialized/3x3Grid2lanes.{type}.xml",
        "begin_time": 0,
        "end_time": 3600,  # TODO: increase to 26000
    },
    "3x3grid-specialized5": {
        "path": "sumo_rl_nets/3x3grid-specialized5/3x3Grid2lanes.{type}.xml",
        "begin_time": 0,
        "end_time": 3600,  # TODO: increase to 26000
    },
    "3x3grid-3lanes": {
        "path": "sumo_rl_nets/3x3grid-3lanes/3x3Grid3lanes.{type}.xml",
        "begin_time": 0,
        "end_time": 3600,  # TODO: increase to 26000
    },
    "3x3grid-3lanes2": {
        "path": "sumo_rl_nets/3x3grid-3lanes2/3x3Grid3lanes.{type}.xml",
        "begin_time": 0,
        "end_time": 3600,  # TODO: increase to 26000
    },
    "3x3grid-3lanes3": {
        "path": "sumo_rl_nets/3x3grid-3lanes3/3x3Grid3lanes.{type}.xml",
        "begin_time": 0,
        "end_time": 3600,  # TODO: increase to 26000
    },
}

# INTELLI_LIGHT_REWARD = {
#     "delay": 3,
#     "waiting_time": 3,
#     "light_switches": 2,
# }

INTELLI_LIGHT_REWARD = {"delay": 3, "waiting_time": 2, "light_switches": 1}
INTELLI_LIGHT_PRCOL_REWARD = {
    "delay": 3,
    "waiting_time": 2,
    "light_switches": 1,
    "out_lanes_availability": 1,
}


CURRICULUM_SETTINGS = {
    "custom-2way-single-intersection3": {"0-1000": 0, "1000-1500": 1, "1500-4000": 0},
    "3x3grid-3lanes2": {
        "0-1000": 0,
        "1000-2800": 1,
        "2800-4000": 0,
    },
}

DYNAMIC_LIGHT_SETTINGS = {
    "3x3grid-3lanes2": {
        "start_time": 1000,
        "end_time": 2800,
    },
}
