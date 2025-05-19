import logging

from agents import base_cyclic
from agents import max_pressure
from agents import sotl
from agents import option_critic_nn
from agents import option_critic_classification
from agents import actor_critic_agent
import stable_baselines3

from sumo_rl_environment.custom_env import CustomSumoEnvironment, BrokenLightEnvironment
from configs import ROUTE_SETTINGS, DYNAMIC_LIGHT_SETTINGS

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete


def setup_env(
    traffic: str,
    reward_fn: str,
    reward_weights: dict = {},
    broken: bool = False,
    broken_mode: str = "full",
    target_model: str = None,
    use_gui: bool = False,
) -> CustomSumoEnvironment:
    """Setup the environment for the given traffic scenario.

    Args:
        traffic (str): traffic to train on

    Returns:
        env: the environment
    """
    settings = ROUTE_SETTINGS[traffic]
    route_file = settings["path"]
    start_time = settings["begin_time"]
    end_time = settings["end_time"]
    duration = end_time - start_time

    # delta_time (int) – Simulation seconds between actions. Default: 5 seconds
    if broken:
        if broken_mode == "full":
            print(
                "Using broken light environment with broken lights for the full duration"
            )
            broken_light_start = None
            broken_light_end = None
        elif broken_mode == "partial":

            broken_light_settings = DYNAMIC_LIGHT_SETTINGS[traffic]
            broken_light_start = broken_light_settings["start_time"]
            broken_light_end = broken_light_settings["end_time"]
            print(
                "Using broken light environment with broken lights for a partial duration "
                f"from {broken_light_start} to {broken_light_end}"
            )
        else:
            raise ValueError(f"Unsupported broken mode {broken_mode}")
        env = BrokenLightEnvironment(
            net_file=route_file.format(type="net"),
            route_file=route_file.format(type="rou"),
            # single_agent=True,
            begin_time=start_time,
            num_seconds=duration,
            reward_fn=reward_fn,
            reward_weights=reward_weights,
            use_gui=use_gui,
            broken_light_start=broken_light_start,
            broken_light_end=broken_light_end,
        )
    else:
        print("Using normal environment")
        env = CustomSumoEnvironment(
            net_file=route_file.format(type="net"),
            route_file=route_file.format(type="rou"),
            # single_agent=True,
            begin_time=start_time,
            num_seconds=duration,
            reward_fn=reward_fn,
            reward_weights=reward_weights,
            use_gui=use_gui,
        )

    print("Environment created")
    if target_model and (
        target_model.startswith("a2c") or target_model.startswith("option_critic")
    ):
        env = DictToFlatActionWrapper(env)
    return env


class DictToFlatActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # Assume the original action space is a Dict of Discrete spaces
        self.keys = list(env.action_space.spaces.keys())
        # Build a MultiDiscrete space from the dict values
        self.action_n = [env.action_space.spaces[key].n for key in self.keys]
        self.action_space = MultiDiscrete(self.action_n)

    def action(self, action):
        # Convert the flat array back into a dict
        return {key: int(a) for key, a in zip(self.keys, action)}

    def __getattr__(self, name):
        """Updated to also be able to directly access the env attributes"""
        try:
            # Avoid recursion: use base implementation to get .inner
            env = super().__getattribute__("env")
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        # Delegate attribute lookup to the inner object
        try:
            return getattr(env, name)
        except AttributeError as e:
            logging.warning(e)
            # Optional: customise the error message
            return super().__getattribute__(name)


def load_model(model: str, env: CustomSumoEnvironment):
    """Load the model based on the model name

    Args:
        model (str): Target model
        env (CustomSumoEnvironment): Environment in which to run the model

    Raises:
        ValueError: unsupported model

    Returns:
        Any: (trained) model
    """
    if model.startswith("cyclic_15"):
        return base_cyclic.CyclicAgent(env=env, green_duration=15)
    elif model.startswith("cyclic_30"):
        return base_cyclic.CyclicAgent(env=env, green_duration=30)
    elif model.startswith("max_pressure"):
        return max_pressure.MaxPressureAgent(env=env)
    elif model.startswith("sotl"):
        return sotl.SOTLPlatoonAgent(env=env)
    elif model.startswith("a2c") or model.startswith("testing_finetuning"):
        agent = stable_baselines3.A2C(
            policy="MultiInputPolicy",
            env=env,
            device="cpu",
        )

        agent = agent.load(
            f"./models/{model}.zip",
        )
        return agent
    elif model.startswith("actor_critic") or model.startswith("testing_finetuning"):
        agent = actor_critic_agent.CustomActorCritic(env=env, device="cpu")
        agent.load(f"./models/{model}")
        return agent
    elif model.startswith("option_critic"):
        split_model = model.split("_")
        for index, item in enumerate(split_model):
            if item == "options":
                num_options = int(split_model[index - 1])
        if model.startswith("option_critic_nn"):
            agent = option_critic_nn.OptionCriticNeuralNetwork(
                env=env,
                num_options=num_options,
                temperature=0.1,
                eps_start=0.9,
                eps_min=0.1,
                eps_decay=0.999,
                eps_test=0.05,
                device="cpu",
            )
        elif model.startswith("option_critic_classification"):
            agent = option_critic_classification.OptionCriticClassification(
                env=env,
                num_options=num_options,
                temperature=0.1,
                eps_start=0.9,
                eps_min=0.1,
                eps_decay=0.999,
                eps_test=0.05,
                device="cpu",
            )
        else:
            raise ValueError(f"Unsupported model {model}")
        agent.load(f"./models/{model}")
        # agent.load_state_dict(
        #     torch.load(
        #         f"./models/{model}",
        #         map_location="cpu"
        #     )["model_params"]
        # )
        return agent
    else:
        raise ValueError("Unsupported model")
