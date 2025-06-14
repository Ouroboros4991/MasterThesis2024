
# import wandb


# import supersuit as ss
import stable_baselines3
from stable_baselines3.common.monitor import Monitor

# from wandb.integration.sb3 import WandbCallback


from sumo_rl_environment.custom_env import CustomSumoEnvironment

from configs import ROUTE_SETTINGS

TRAFFIC = "custom-2way-single-intersection"  # "cologne1"
SETTINGS = ROUTE_SETTINGS[TRAFFIC]
N_EPISODES = 100


def main():
    route_file = SETTINGS["path"]
    start_time = SETTINGS["begin_time"]
    end_time = SETTINGS["end_time"]
    duration = end_time - start_time
    experiment_name = f"dqn_{TRAFFIC}"
    # delta_time (int) – Simulation seconds between actions. Default: 5 seconds
    env = CustomSumoEnvironment(
        net_file=route_file.format(type="net"),
        route_file=route_file.format(type="rou"),
        # single_agent=True,
        begin_time=start_time,
        num_seconds=duration,
    )
    print("Environment created")

    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class="stable_baselines3")
    # env = VecMonitor(env)

    env = Monitor(env)

    env.reset()

    agent = stable_baselines3.DQN(
        "MlpPolicy",
        env,
        verbose=3,
        gamma=0.95,
        batch_size=256,
        tensorboard_log=f"runs/{experiment_name}",
    )
    agent.learn(total_timesteps=100000)
    agent.save(f"./models/{experiment_name}")


if __name__ == "__main__":
    main()
