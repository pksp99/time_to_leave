import sys
import os
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

import logging

# Configure the logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


alpha_range = [2, 3, 4, 6]
beta_range = [1, 2, 3, 4]
h_range = [1/15, 2/15, 10/15]
c_range = [20, 25, 30, 10]
N_range = range(3, 40)
n_range = [5]

def process_iter(id):
    dic = dict()

    alpha = random.choice(alpha_range)
    dic['alpha'] = alpha

    beta = random.choice(beta_range)
    dic['beta'] = beta

    h = random.choice(h_range)
    dic['h'] = h

    c = random.choice(c_range)
    dic['c'] = c

    N = random.choice(N_range)
    dic['N'] = N

    n = random.choice(n_range)
    dic['n'] = n

    intervals = np.random.gamma(shape=alpha, scale=beta, size=N+n)

    mean_n = statistics.mean(intervals[:n])
    dic['mean_n'] = mean_n

    std_n = statistics.stdev(intervals[:n])
    dic['std_n'] = std_n

    alpha_hat, beta_hat = methods.gamma_estimate_parameters(n, intervals)
    alpha_hat = round(alpha_hat) # fractional alpha has no solution for u*, hencing rounding it.
    dic['alpha_hat'], dic['beta_hat'] = alpha_hat, beta_hat

    logging.info(f"Start process-{id}: {dic}")
    if(alpha_hat < 2 or alpha_hat > 20):
        logging.critical(f'End process-{id}: unable to compute u* or alpha_hat is too large')
        return 'Nil'

    u = methods.cal_actual_time(n, intervals)
    dic['u'] = u

    u_star = methods.get_u_star_binary(N, alpha, beta, h, c)
    dic['u_star'] = u_star

    u_star_hat = methods.get_u_star_binary(N, alpha_hat, beta_hat, h, c)
    dic['u_star_hat'] = u_star_hat

    z=u_star / u_star_hat
    dic['z'] = z

    optimal_cost = methods.cal_cost(c, h, u, u_star)
    dic['optimal_cost'] = optimal_cost

    actual_cost = methods.cal_cost(c, h, u, u_star_hat)
    dic['actual_cost'] = actual_cost

    intervals_str = '_'.join([str(x) for x in intervals])
    dic['intervals_str'] = intervals_str

    logging.info(f'End process-{id}: {dic}')
    return dic

import multiprocessing
from tqdm import tqdm

def update_progress(_):
    global pbar
    pbar.update()


def main():
    global pbar
    n = 5000
    pbar = tqdm(total=n, desc="Processing")

    with multiprocessing.Pool() as pool:
        results = []
        for result in pool.imap_unordered(process_iter, range(n)):
            update_progress(result)
            results.append(result)

    pbar.close()
    results = [x for x in results if x != 'Nil']
    df = pd.DataFrame(results)
    df.to_csv('gen_data.csv', index=False)

if __name__ == '__main__':

    file_name = 'app.log'
    if os.path.exists(file_name):
        os.remove(file_name)
    main()