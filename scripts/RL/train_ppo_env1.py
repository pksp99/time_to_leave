import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils import methods
from utils.rl_environments.env1 import Env1

import sys

CONSTANT_CONFIG = {
    'alpha_range': [2],
    'beta_range': [1],
    'h_range': [0.15],
    'c_range': [25],
    'total': range(10, 40), 
}


PPO_MODEL_NAME = "PPO_ENV1"
if __name__ == '__main__':

    if len(sys.argv) > 1:
        total_timesteps = int(sys.argv[1])
    else:
        total_timesteps = 1_000_000

    n_cpus = os.cpu_count()

    print(f"Number of processors: {n_cpus}")
    env = make_vec_env(lambda: Env1(config=CONSTANT_CONFIG), n_envs=n_cpus, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env, verbose=1, device='cpu')

    print(f"Training model for {total_timesteps} timesteps")
    model.learn(total_timesteps=total_timesteps)

    model.save(methods.file_path(PPO_MODEL_NAME, 'models'))


