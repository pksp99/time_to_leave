import concurrent.futures
import logging
import os
import random
import statistics

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import methods


def process_iter(config, log_file, id):
    # Configure the logging
    logging.basicConfig(
        filename=file_path(log_file),
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    dic = dict()

    alpha = random.choice(config['alpha_range'])
    dic['alpha'] = alpha

    beta = random.choice(config['beta_range'])
    dic['beta'] = beta

    h = random.choice(config['h_range'])
    dic['h'] = h

    c = random.choice(config['c_range'])
    dic['c'] = c

    N = random.choice(config['N_range'])
    dic['N'] = N

    n = random.choice(config['n_range'])
    dic['n'] = n

    intervals = np.random.gamma(shape=alpha, scale=beta, size=N + n)

    mean_n = statistics.mean(intervals[:n])
    dic['mean_n'] = mean_n

    std_n = statistics.stdev(intervals[:n])
    dic['std_n'] = std_n

    alpha_hat, beta_hat = methods.gamma_estimate_parameters(n, intervals)
    dic['alpha_hat'], dic['beta_hat'] = alpha_hat, beta_hat

    intervals_str = '_'.join([str(x) for x in intervals])
    dic['intervals_str'] = intervals_str

    logging.info(f"Start process-{id}: {dic}")

    # Unable to compute u* cases
    if alpha_hat * N <= 1:
        logging.critical(f'End process-{id}: alpha_hat < 1')
        return 'Nil'
    elif h / c >= 1 / beta:
        logging.critical(f'End process-{id}: Impossible beta: {beta}, for h: {h} and c: {c}')
        return 'Nil'
    elif h / c >= 1 / beta_hat:
        logging.critical(f'End process-{id}: Impossible beta_hat: {beta_hat}, for h: {h} and c: {c}')
        return 'Nil'

    u = methods.cal_actual_time(n, intervals)
    dic['u'] = u

    u_star = methods.get_u_star_binary_fast(N, alpha, beta, h, c)
    dic['u_star'] = u_star

    u_star_hat = methods.get_u_star_binary_fast(N, alpha_hat, beta_hat, h, c)
    dic['u_star_hat'] = u_star_hat

    z = u_star / u_star_hat
    dic['z'] = z

    optimal_cost = methods.cal_cost(c, h, u, u_star)
    dic['optimal_cost'] = optimal_cost

    actual_cost = methods.cal_cost(c, h, u, u_star_hat)
    dic['actual_cost'] = actual_cost

    logging.info(f'End process-{id}: {dic}')
    return dic


def update_progress(_):
    global pbar
    pbar.update()


def file_path(file_name):
    # Get the directory of the current module
    module_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the sibling directory path
    parent_dir = os.path.dirname(module_dir)
    data_folder_path = os.path.join(parent_dir, 'data')

    # Create the full file path
    file_path = os.path.join(data_folder_path, file_name)
    return file_path


def generate(config: dict, output: str, log_file: str, n: int = 1000):
    # Delete log file if exists
    if os.path.exists(file_path(log_file)):
        os.remove(file_path(log_file))

    global pbar
    pbar = tqdm(total=n, desc="Processing")

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        submits = [executor.submit(process_iter, config, log_file, i) for i in range(n)]
        for f in concurrent.futures.as_completed(submits):
            update_progress(f.result())
            results.append(f.result())

    pbar.close()
    results = [x for x in results if x != 'Nil']
    df = pd.DataFrame(results)
    df.to_csv(file_path(output), index=False)
