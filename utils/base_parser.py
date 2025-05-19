import argparse

import configs


def generate_base_parser():
    """Generate the base parser for the command line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate the provided model using the given traffic scenario.",
    )
    parser.add_argument(
        "-t", "--traffic", choices=configs.ROUTE_SETTINGS.keys(), required=True
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=100000,
        help="Number of steps to train the agent.",
    )
    parser.add_argument(
        "-b", "--broken", action="store_true", help="Use broken traffic light scenario."
    )
    parser.add_argument(
        "-bm",
        "--broken-mode",
        type=str,
        help="Define broken mode. Defaults to full broken mode",
        default="full",
    )
    parser.add_argument(
        "-r",
        "--reward_fn",
        type=str,
        help="Reward function to use",
        default="intelli_light_reward",
    )
    return parser
