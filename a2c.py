# import wandb


# import supersuit as ss
import stable_baselines3
from stable_baselines3.common.monitor import Monitor

# from wandb.integration.sb3 import WandbCallback


from utils import utils
from utils import base_parser


def train(
    env,
    traffic: str,
    steps: int = 30000,
    reward_fn: str = "pressure",
    broken: bool = False,
):

    if broken:
        experiment_name = f"a2c_broken_{traffic}_{steps}_steps"
    else:
        experiment_name = f"a2c_{traffic}_{steps}_steps"
    experiment_name += reward_fn
    if env.env.reward_weights:
        experiment_name += "_" + "_".join(
            [f"{key}_{value}" for key, value in env.env.reward_weights.items()]
        )
    # env = DummyVecEnv([lambda: env])
    env.reset()

    agent = stable_baselines3.A2C(
        "MultiInputPolicy",
        env,
        verbose=3,
        gamma=0.95,
        tensorboard_log=f"runs/{experiment_name}",
        # learning_rate=0.0001,
        ent_coef=0.02,
    )

    agent.learn(total_timesteps=steps)
    agent.save(f"models/{experiment_name}.zip")
    return agent


def main(traffic: str, steps: int, reward_fn: str, broken: bool = False):
    """Main function to train the agent."""

    env = utils.setup_env(
        traffic,
        reward_fn,
        broken=broken,
        target_model="a2c",
    )
    env = Monitor(env)

    train(env, traffic, steps, reward_fn, broken)


if __name__ == "__main__":
    parser = base_parser.generate_base_parser()
    args = parser.parse_args()
    # reward_fn = "intelli_light_prcol_reward"

    main(args.traffic, args.steps, args.reward_fn, args.broken)
