import argparse


from configs import ROUTE_SETTINGS
from utils import utils


def visualize(
    traffic: str, model: str, broken: bool = False, broken_mode: str = "full"
):
    env = utils.setup_env(
        traffic,
        # reward_fn="pressure",
        reward_fn="intelli_light_prcol_reward",
        broken=broken,
        target_model=model,
        use_gui=True,
        broken_mode=broken_mode,
    )
    obs, _ = env.reset()

    agent = utils.load_model(model, env)

    terminate = False
    while not terminate:
        try:
            action_dict, _states = agent.predict(obs)
        except AttributeError:
            # Option critic
            obs = agent._convert_dict_to_tensor(obs)
            state = agent.prep_state(obs)
            # state = agent.prep_state(obs)
            action_dict, additional_info = agent.get_action(state)
            # action_dict = agent.convert_action_to_dict(action)
        # print("Action dict:", action_dict)
        obs, reward, terminate, truncated, info = env.step(action_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the provided model using the given traffic scenario.",
    )
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-t", "--traffic", choices=ROUTE_SETTINGS.keys(), required=True)
    parser.add_argument(
        "-b",
        "--broken",
        action="store_true",
        help="Use a broken traffic light environment.",
    )
    parser.add_argument(
        "-bm",
        "--broken-mode",
        type=str,
        help="Define broken mode. Defaults to full broken mode",
        default="full",
    )
    args = parser.parse_args()
    visualize(args.traffic, args.model, args.broken, args.broken_mode)
