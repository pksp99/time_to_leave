import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Sequence, Dict
import numpy as np
from utils import methods
import statistics
from utils.rl_environments import GammaIntervalsEnv1


CONFIG = {
    'alpha_range': range(2, 8),
    'beta_range': [round(i * 0.5, 1) for i in range(2, 9)],
    'h_range': [round(i * 0.01, 2) for i in range(6, 61)],
    'c_range': range(20, 30),
    'total': range(10, 40), 
}


from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

env = GammaIntervalsEnv1()
env.reset()


model = DQN("MlpPolicy", env, verbose=1, device='cpu', learning_rate=0.00001)


model.learn(total_timesteps=20000)



model.save("dqn")

model = DQN.load("dqn")


state, info = env.reset()
done = False
total_reward = 0
print(info)

while not done:
    action, _ = model.predict(state)  
    state, reward, done, _, _ = env.step(action)
    total_reward += reward
    env.render()

print("Total reward:", total_reward)