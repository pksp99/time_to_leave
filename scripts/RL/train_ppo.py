import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils import methods
from utils.rl_environments.env2 import Env2

import sys

PPO_MODEL_NAME = "PPO_SC"
if __name__ == '__main__':

    if len(sys.argv) > 1:
        total_timesteps = int(sys.argv[1])
    else:
        total_timesteps = 2000

    n_cpus = os.cpu_count()

    print(f"Number of processors: {n_cpus}")
    env = make_vec_env(Env2, n_envs=n_cpus, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env, verbose=1, device='cpu')

    print(f"Training model for {total_timesteps} timesteps")
    model.learn(total_timesteps=total_timesteps)

    model.save(methods.file_path(PPO_MODEL_NAME, 'models'))


