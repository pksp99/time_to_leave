import numpy as np

from utils import methods
from utils.rl_environments.env2 import Env2
import pandas as pd
import sys

PPO_MODEL_NAME = "PPO_SC"

if len(sys.argv) > 1:
    total_test = int(sys.argv[1])
else:
    total_test = 30

def get_realized_data():
    flag = False
    while not flag:
        alpha = np.random.exponential(scale=8.82) + 1.02
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



df = pd.DataFrame(columns=['h', 'c', 'travel_time', 'total', 'intervals', 'alpha', 'beta'])

for i in range(total_test):
    alpha, beta, h, c, total, intervals, travel_time = get_realized_data()
    row = {'h': h, 'c': c, 'travel_time': travel_time, 'total': total, 'intervals': intervals, 'alpha': alpha,
           'beta': beta}
    df.loc[i] = row

from stable_baselines3 import PPO

model = PPO.load(methods.file_path(PPO_MODEL_NAME, 'models'))

env = Env2()
rewards = {}
u_rl = {}
print(env._get_info())
for i in range(len(df)):
    row = df.iloc[i]
    print(row)
    state, _ = env.reset(row=row)
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(state)
        state, reward, done, _, info = env.step(action)
        total_reward += reward
    rewards[i] = total_reward
    u_rl[i] = env.cur_time + env.travel_time
    # print(f"Episode {i} reward: {total_reward} | h = {info['state']['h']}, c = {info['state']['c']}")
    print(f"Episode {i} reward: {total_reward} | cur_time = {info['state']['cur_time']}")

df['rewards'] = rewards
df['u_rl'] = u_rl
print(f"Average reward: {df['rewards'].mean()}")

from pandarallel import pandarallel

# Initialize pandarallel with progress bar enabled
pandarallel.initialize(progress_bar=False)
for i in [0, 3, 5]:
    print(f"Optimal reward at n = {i}")
    df[f'u{i}'] = df.apply(lambda row: row['intervals'][i:].sum(), axis=1)
    df[f'u_star{i}'] = df.parallel_apply(
        lambda row: methods.get_u_star_binary_fast(row['total'] - i, row['alpha'], row['beta'], row['h'], row['c']),
        axis=1)
    df[f'optimal_rewards{i}'] = df.apply(
        lambda row: -methods.cal_cost(row['c'], row['h'], row[f'u{i}'], row[f'u_star{i}']), axis=1)
    
df['direct_leave_rewards'] = df.apply(lambda row: -methods.cal_cost(row['c'], row['h'], row['intervals'][3:].sum(), row[f'travel_time']), axis=1)

df.to_csv(methods.file_path('ppo_df.csv', 'data'), index=False)
