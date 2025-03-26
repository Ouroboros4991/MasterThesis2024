
import os
import torch

from agents import base_cyclic
from agents import max_pressure
from agents import sotl
from agents import option_critic
from agents import option_critic_nn
from agents import option_critic_classification
from agents import actor_critic_agent

from sumo_rl_environment.custom_env import CustomSumoEnvironment
from configs import ROUTE_SETTINGS


def create_env(traffic: str):
    """Create the environment based on the traffic setting"""
    os.environ["LIBSUMO_AS_TRACI"] = "1" 
    os.environ["SUMO_HOME"] = "/usr/share/sumo"       
    
    settings = ROUTE_SETTINGS[traffic]

    route_file = settings["path"]
    start_time = settings["begin_time"]
    end_time = settings["end_time"]
    duration = end_time - start_time
    env = CustomSumoEnvironment(
        net_file=route_file.format(type="net"),
        route_file=route_file.format(type="rou"),
        # single_agent=True,
        begin_time=start_time,
        num_seconds=duration,
    )
    env.reset()
    return env

def load_model(model: str, env:CustomSumoEnvironment):
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
    elif model.startswith("actor_critic"):
        agent = actor_critic_agent.CustomActorCritic(env=env, device="cpu")
        agent.load(
            f"./models/{model}"
        )
        return agent
    elif model.startswith("option_critic"):
        split_model = model.split("_")
        for index, item in enumerate(split_model):
            if item == "options":
                num_options = int(split_model[index -1])
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
        agent.load(
            f"./models/{model}"
        )
        # agent.load_state_dict(
        #     torch.load(
        #         f"./models/{model}",
        #         map_location="cpu"
        #     )["model_params"]
        # )
        return agent
    else:
        raise ValueError("Unsupported model")