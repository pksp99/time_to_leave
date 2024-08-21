import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from utils import math_expressions as mexpr, methods, generate_data

import mlflow

import mulitple_models_v1 as mm


# Configuration for data generation
TRAINING_CONFIG = {
    'alpha_range': [2, 4, 6, 10],
    'beta_range': [round(i * 0.5, 1) for i in range(2, 9)],
    'h_range': [round(i * 0.01, 2) for i in range(6, 61)],
    'c_range': range(20, 30),
    'N_range': range(10, 40),
    'n_range': [5]
}

TEST_CONFIG = {
    'alpha_range': [3, 5, 7, 8],
    'beta_range': [1, 1.5, 2, 3],
    'h_range': [1 / 15],
    'c_range': [25],
    'N_range': range(15, 40),
    'n_range': [5]
}

EXPERIMENT = 'Delibrate_diff'

TOTAL = 20000

gen_test_file = 'gen_test_v2.csv'
gen_train_file = 'gen_train_v2.csv'

     

def main():
    generate_data.generate(TEST_CONFIG,
                            gen_test_file,
                            'app.log',
                            n=TOTAL)

    generate_data.generate(TRAINING_CONFIG,
                            gen_train_file,
                            'app.log',
                            n=TOTAL)

    mm.experiment(gen_test_file=gen_test_file, gen_train_file=gen_train_file, experiment_name=EXPERIMENT)

    def generate_custom_n(input:str, n:int):
        df = pd.read_csv(methods.file_path(input))
        total = df['n'] + df['N']
        df['n'] = n
        df['N'] = total - df['n']
        generate_data.generate(config=dict(),
                            output=str(n)+input,
                            log_file='app.log',
                            df=df)

    for n in [6, 8, 10, 12]:
        n_gen_test_file = str(n) + gen_test_file
        n_gen_train_file = str(n) + gen_train_file

        generate_custom_n(gen_test_file, n)
        generate_custom_n(gen_train_file, n)

        mm.experiment(gen_test_file=n_gen_test_file, gen_train_file=n_gen_train_file, experiment_name=str(n) + EXPERIMENT)

if __name__ == '__main__':
    main()


