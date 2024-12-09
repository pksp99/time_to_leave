import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import statistics
from utils import methods
import numpy as np

# Internally not using gamma distribution

DEFAULT_CONFIG = {
    'alpha_range': range(1, 8),
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
    # travel_time = sum(intervals[3:]) - np.random.exponential(scale=beta)
    # travel_time = sum(intervals[3:]) - np.random.gamma(shape=2, scale=alpha*beta)
    # travel_time = max(alpha * beta * 2, travel_time)
    travel_time = sum(intervals[3:]) * np.random.uniform(0, 1)
    travel_time = max(alpha * beta, travel_time)

    return alpha, beta, h, c, total, intervals, travel_time

class Env3(gym.Env):
    def __init__(self, step_size=0.1, config=DEFAULT_CONFIG):
        super(Env3, self).__init__()

        self.config = config
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
        self.last_update = -1

        # 0 = wait, 1 = leave
        # self.action_space = Discrete(2)
        self.action_space = Discrete(4)

        obs_dim = 9
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
                'last_update': self.last_update
            }
        }
    def cal_derived_data(self):
        self.N = self.total - self.n
        self.obs_intervals = self.intervals[:self.n]
        self.mean_n = statistics.mean(self.obs_intervals)
        self.std_n = statistics.stdev(self.obs_intervals)
        self.last_update = self.cum_sum_intervals[self.n - 1]
        

    def reset(self, seed=None, options=None, config=None, row=None):

        if config is None:
            config = self.config

        if row is not None:
            self.h = row['h']
            self.c = row['c']
            self.total = int(row['total'])
            self.travel_time = row['travel_time']
            if isinstance(row['intervals'], list):
                self.intervals = row['intervals']
            else:
                self.intervals = row['intervals'].tolist()

        else:
            self.alpha, self.beta, self.h, self.c, self.total, self.intervals, self.travel_time = get_realized_data(config)
        
        self.cum_sum_intervals = np.cumsum(self.intervals)

        self.n =  3

        self.cur_time = self.cum_sum_intervals[self.n - 1]

        self.cal_derived_data()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.cur_time += self.step_size

        if self.cur_time >= self.cum_sum_intervals[-1]:
            action = 1

        if action == 1:
            cost = methods.cal_cost(c=self.c, h=self.h, actual_time=self.cum_sum_intervals[-1], predicted_time=self.cur_time + self.travel_time)
            self.obs_intervals = self.intervals[:self.total]
            self.n = self.total
            self.N = 0
            return self._get_obs(), -cost, True, False, self._get_info()
        else:
            while self.cur_time >= self.cum_sum_intervals[self.n]:
                self.n += 1
                self.cal_derived_data()
            return self._get_obs(), 0, False, False, self._get_info()

    def render(self, mode='human'):
        print(self._get_obs())