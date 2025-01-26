from agents import base_cyclic
from agents import max_pressure
from agents import sotl
from agents import option_critic
from agents import option_critic_nn

from sumo_rl_environment.custom_env import CustomSumoEnvironment

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
    else:
        raise ValueError("Unsupported model")