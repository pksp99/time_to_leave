import sys
from pathlib import Path
parent_dir = Path('.').resolve().parent
sys.path.insert(0, str(parent_dir))

import numpy as np
import pandas as pd
import random
import statistics
import importlib

import math_expressions as mexpr
importlib.reload(mexpr)

import methods
importlib.reload(methods)


alpha_range = [2, 3, 4, 6, 8]
beta_range = [1, 2, 4, 6]
h_range = [1/15, 2/15, 10/15]
c_range = [20, 25, 30, 10]
N_range = range(3, 60)
n_range = [5]

def process_iter(_):
    alpha = random.choice(alpha_range)
    beta = random.choice(beta_range)
    h = random.choice(h_range)
    c = random.choice(c_range)
    N = random.choice(N_range)
    n = random.choice(n_range)
    intervals = np.random.gamma(shape=alpha, scale=beta, size=N+n)

    mean_n = statistics.mean(intervals[:n])
    std_n = statistics.stdev(intervals[:n])

    alpha_hat, beta_hat = methods.gamma_estimate_parameters(n, intervals)
    alpha_hat = round(alpha_hat) # fractional alpha has no solution for u*, hencing rounding it.

    u = methods.cal_actual_time(n, intervals)
    u_star = methods.get_u_star_binary(N, alpha, beta, h, c)
    u_star_hat = methods.get_u_star_binary(N, alpha_hat, beta_hat, h, c)
    z=u_star / u_star_hat

    optimal_cost = methods.cal_cost(c, h, u, u_star)
    actual_cost = methods.cal_cost(c, h, u, u_star_hat)

    intervals_str = '_'.join([str(x) for x in intervals])
    dic = {'alpha': alpha,
           'beta': beta,
           'N': N,
           'n':n,
           'h': h,
           'c': c,
           'mean_n': mean_n,
           'std_n': std_n,
           'alpha_hat': alpha_hat,
           'beta_hat': beta_hat,
           'u_star': u_star,
           'u_star_hat': u_star_hat,
           'z': z,
           'u': u,
           'optimal_cost': optimal_cost,
           'actual_cost': actual_cost,
           'intervals_str': intervals_str}
    return dic

import multiprocessing
from tqdm import tqdm

def update_progress(_):
    global pbar
    pbar.update()


def main():
    global pbar
    n = 100
    pbar = tqdm(total=n, desc="Processing")

    with multiprocessing.Pool() as pool:
        results = []
        for result in pool.imap_unordered(process_iter, range(n)):
            update_progress(result)
            results.append(result)

    pbar.close()
    df = pd.DataFrame(results)
    df.to_csv('gen_data.csv', index=False)

if __name__ == '__main__':
    main()