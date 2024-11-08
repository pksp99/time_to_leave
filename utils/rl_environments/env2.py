import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import statistics
from utils import methods
import numpy as np



def get_realized_data():
    flag = False
    while not flag:
        alpha = np.random.exponential(scale=8.82)
        beta = np.random.exponential(1.94)
        h = np.random.exponential(1.30)
        c = np.random.exponential(4)
        flag = alpha and beta and h and c and (h / c < 1 / beta)
    total = int(np.random.normal(40, 6.3))
    intervals = np.random.gamma(shape=alpha, scale=beta, size=total)
    # travel_time = sum(intervals[3:]) - np.random.exponential(scale=beta)
    travel_time = sum(intervals[3:]) - np.random.gamma(shape=2, scale=alpha * beta)
    travel_time = max(alpha * beta * 2, travel_time)

    return alpha, beta, h, c, total, intervals, travel_time


class Env2(gym.Env):
    def __init__(self, step_size=0.1):
        super(Env2, self).__init__()

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

        # Unable to compute u* cases
        if self.alpha_hat * self.N <= 1 or self.alpha_hat * self.N > 600:
            return
        elif self.h / self.c >= 1 / self.beta:
            return
        elif self.h / self.c >= 1 / self.beta_hat:
            return
        self.u_star_hat = methods.get_u_star_binary_fast(self.N, self.alpha_hat, self.beta_hat, self.h, self.c)

    def reset(self, seed=None, options=None, row=None):

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
            self.alpha, self.beta, self.h, self.c, self.total, self.intervals, self.travel_time = get_realized_data()

        self.cum_sum_intervals = np.cumsum(self.intervals)

        self.n = 3

        self.cur_time = self.cum_sum_intervals[self.n - 1]

        self.cal_derived_data()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.cur_time += self.step_size

        if self.cur_time >= self.cum_sum_intervals[-1]:
            action = 1

        if action == 1:
            cost = methods.cal_cost(c=self.c, h=self.h, actual_time=self.cum_sum_intervals[-1],
                                    predicted_time=self.cur_time + self.travel_time)
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