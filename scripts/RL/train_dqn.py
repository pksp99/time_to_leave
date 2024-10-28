import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Sequence, Dict
import numpy as np
from utils import methods
import statistics


CONFIG = {
    'alpha_range': range(2, 8),
    'beta_range': [round(i * 0.5, 1) for i in range(2, 9)],
    'h_range': [round(i * 0.01, 2) for i in range(6, 61)],
    'c_range': range(20, 30),
    'total': range(10, 40), 
}

def get_realized_data(config):
    alpha = np.random.choice(config['alpha_range'])
    beta = np.random.choice(config['beta_range'])
    h = np.random.choice(config['h_range'])
    c = np.random.choice(config['c_range'])
    total = np.random.choice(config['total'])
    intervals = np.random.gamma(shape=alpha, scale=beta, size=total)
    travel_time = sum(intervals[4:]) - np.random.exponential(scale=beta)
    travel_time = max(beta * 2, travel_time)

    return alpha, beta, h, c, total, intervals, travel_time

class CustomEnv(gym.Env):
    def __init__(self, step_size=1):
        super(CustomEnv, self).__init__()

        self.alpha = -1
        self.beta = -1
        self.h = -1
        self.c = -1
        self.total = -1
        self.intervals = -1
        self.travel_time = -1
        self.cur_time = -1
        self.obs_intervals = -1
        self.n = -1
        self.N = -1
        self.cum_sum_intervals = -1

        self.step_size = step_size


        self.mean_n = -1
        self.std_n = -1
        self.alpha_hat, self.beta_hat = -1, -1
        self.u_star_hat = -1
        self.last_update = -1

        # 0 = wait, 1 = leave
        self.action_space = Discrete(2)

        obs_dim = 12
        self.observation_space = Box(low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    
    def _get_obs(self):
        return np.array([
            self.h,
            self.c,
            self.n,
            self.N,
            self.travel_time,
            self.cur_time,
            self.mean_n,
            self.std_n,
            self.alpha_hat,
            self.beta_hat,
            self.u_star_hat,
            self.last_update
        ], dtype=np.float32)
    
    
    def _get_info(self):

        return {
            'hidden': {
                'alpha': self.alpha,
                'beta': self.beta,
                'interval': self.intervals,
                'cum_sum_intervals': self.cum_sum_intervals,
            },
            'state': {
                'n': self.n,
                'N': self.N,
                'h': self.h,
                'c': self.c,
                'travel_time': self.travel_time,
                'cur_time': self.cur_time,
                'mean_n': self.mean_n,
                'std_n': self.std_n,
                'alpha_hat': self.alpha_hat,
                'beta_hat': self.beta_hat,
                'u_star_hat': self.u_star_hat,
                'last_update': self.last_update
            }
        }
    def cal_derived_data(self):
        self.N = self.total - self.n
        self.obs_intervals = self.intervals[:self.n]
        self.mean_n = statistics.mean(self.obs_intervals)
        self.std_n = statistics.stdev(self.obs_intervals)
        self.alpha_hat, self.beta_hat = methods.gamma_estimate_parameters(self.n, self.intervals)
        self.last_update = self.cum_sum_intervals[self.n - 1]
        self.u_star_hat = methods.get_u_star_binary_fast(self.N, self.alpha_hat, self.beta_hat, self.h, self.c)

    def reset(self, seed=None, options=None):

        self.alpha, self.beta, self.h, self.c, self.total, self.intervals, self.travel_time = get_realized_data(CONFIG)
        
        self.cum_sum_intervals = np.cumsum(self.intervals)

        self.n =  3

        self.cur_time = self.cum_sum_intervals[self.n - 1]

        self.cal_derived_data()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.cur_time += self.step_size

        if self.cur_time >= self.cum_sum_intervals[-1]:
            action = 1

        if action == 0:
            while self.cur_time >= self.cum_sum_intervals[self.n]:
                self.n += 1
                self.cal_derived_data()
            return self._get_obs(), 0, False, False, self._get_info()
        else:
            cost = methods.cal_cost(c=self.c, h=self.h, actual_time=self.cum_sum_intervals[-1], predicted_time=self.cur_time + self.travel_time)
            self.obs_intervals = self.intervals[:int(self.total)]
            self.n = self.total
            self.N = 0
            return self._get_obs(), -cost, True, False, self._get_info() 

    def render(self, mode='human'):
        print(self._get_obs())

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

env = CustomEnv()
env.reset()


model = DQN("MlpPolicy", env, learning_rate=0.001, verbose=1)


model.learn(total_timesteps=100)



model.save("dqn")

model = DQN.load("dqn")


state, _ = env.reset()
done = False
total_reward = 0
model.predict(state)

while not done:
    action, _ = model.predict(state)  
    state, reward, done, _, _ = env.step(action)
    total_reward += reward
    env.render()

print("Total reward:", total_reward)