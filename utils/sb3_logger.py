# Based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/logger.py#L21
import time

from stable_baselines3.common import utils
from stable_baselines3.common.utils import safe_mean


class SB3Logger():
    def __init__(self,
                 verbose: int,
                 tensorboard_log: str,
                 tb_log_name: str = "run",
                 reset_num_timesteps: bool = True,):
        self.logger = utils.configure_logger(verbose, tensorboard_log, tb_log_name, reset_num_timesteps)
        self.episode_start_time = time.time()
        self.ep_info_buffer = {
            'reward': [],
            'length': [],
        }
    
    def start_episode(self, num_timesteps: int):
        self.episode_start_time = time.time()
        self._num_timesteps_at_start = num_timesteps
    
    
    def log_episode(self,
                    num_timesteps: int,
                    iteration: int,
                    reward: float,
                    option_lengths: dict,):
        time_elapsed = (time.time() - self.episode_start_time)
        episode_length = num_timesteps - self._num_timesteps_at_start
        fps = int(episode_length / time_elapsed)

        self.ep_info_buffer['reward'].append(reward)
        self.ep_info_buffer['length'].append(episode_length)
        
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer['reward']) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean(self.ep_info_buffer['reward']))
            self.logger.record("rollout/ep_len_mean", safe_mean(self.ep_info_buffer['length']))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", num_timesteps, exclude="tensorboard")
        
        for option, lengths in option_lengths.items():
            self.logger.record(f"options/option_{option}_mean_length", safe_mean(lengths))
        self.logger.dump(step=num_timesteps)