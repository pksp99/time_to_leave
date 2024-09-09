import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import LinearCustomRegressor

from utils import math_expressions as mexpr, methods, generate_data

import mlflow


# Configuration for data generation
TRAINING_CONFIG = {
    'alpha_range': range(2, 8),
    'beta_range': [round(i * 0.5, 1) for i in range(2, 9)],
    'h_range': [round(i * 0.01, 2) for i in range(6, 61)],
    'c_range': range(20, 30),
    'N_range': range(10, 40),
    'n_range': [5]
}

TEST_CONFIG = {
    'alpha_range': range(2, 8),
    'beta_range': [round(i * 0.5, 1) for i in range(2, 9)],
    'h_range': [round(i * 0.01, 2) for i in range(6, 61)],
    'c_range': range(20, 30),
    'N_range': range(10, 40),
    'n_range': [5]
}

EXPERIMENT = 'Random_same'

TOTAL = 20000

gen_test_file = 'gen_test_v1.csv'
gen_train_file = 'gen_train_v1.csv'


def get_actual_cost_col(df: pd.DataFrame):
    return df.apply(lambda row: methods.cal_cost(row['c'], row['h'], row['u'], row['predicted_u_star']), axis=1)

def mean_absolute_error_percent(actual, predicted):
    
    mape = (abs((actual - predicted) / actual)).mean() * 100
    return mape



def experiment(gen_train_file, gen_test_file, experiment_name):

    train_df = pd.read_csv(methods.file_path(gen_train_file))
    test_df = pd.read_csv(methods.file_path(gen_test_file))

    models = {'linear': LinearRegression(),
                'random_forest':RandomForestRegressor(random_state=50, max_features='sqrt', n_estimators=200, min_samples_leaf=2),
                'gradient_boost': GradientBoostingRegressor(random_state=50, min_samples_split=6, min_samples_leaf=2, max_depth=5)
            }

    reports = []

    # Optimal Model
    metrics = {'mean_cost': test_df['optimal_cost'].mean(),
               'median_cost': test_df['optimal_cost'].median(),
               'esti_time_dif_percent': mean_absolute_error_percent(test_df['u'], test_df['u_star'])}
    reports.append(('Optimal', None, metrics))


    # Average Model
    test_df['predicted_u_star'] = test_df['N'] * test_df['mean_n']
    actual_cost_col = get_actual_cost_col(test_df)
    metrics = {'mean_cost': actual_cost_col.mean(),
                'median_cost': actual_cost_col.median(),
                'esti_time_dif_percent': mean_absolute_error_percent(test_df['u'], test_df['predicted_u_star'])}
    reports.append(('Average', None, metrics))

    # u_star_hat estimate Model
    test_df['predicted_u_star'] = test_df['u_star_hat']
    actual_cost_col = get_actual_cost_col(test_df)
    metrics = {'mean_cost': actual_cost_col.mean(),
                'median_cost': actual_cost_col.median(),
                'esti_time_dif_percent': mean_absolute_error_percent(test_df['u'], test_df['predicted_u_star'])}
    reports.append(('u_star_hat_estimate', None, metrics))

    # linear custom model
    l_model = LinearCustomRegressor.MLModel(train_df, test_df)
    l_model.train()
    test_df['predicted_u_star'] = l_model.get_predictions()
    actual_cost_col = get_actual_cost_col(test_df)
    metrics = {'mean_cost': actual_cost_col.mean(),
                'median_cost': actual_cost_col.median(),
                'esti_time_dif_percent': mean_absolute_error_percent(test_df['u'], test_df['predicted_u_star'])}
    reports.append(('custom_linear', None, metrics))

    for model_name, model in models.items():
        X_train = train_df[['N', 'n', 'h', 'c', 'mean_n', 'std_n', 'alpha_hat', 'beta_hat', 'u_star_hat']]
        y_train = train_df['u_star']

        X_test = test_df[['N', 'n', 'h', 'c', 'mean_n', 'std_n', 'alpha_hat', 'beta_hat', 'u_star_hat']]
        y_test = test_df['u_star']

        model.fit(X_train, y_train)

        test_df['predicted_u_star'] = model.predict(X_test)

        actual_cost_col = get_actual_cost_col(test_df)

        metrics = {'mean_cost': actual_cost_col.mean(),
                'median_cost': actual_cost_col.median(),
                'esti_time_dif_percent': mean_absolute_error_percent(test_df['u'], test_df['predicted_u_star']),
                'train_score': model.score(X_train, y_train),
                'test_score': model.score(X_test, y_test)}
        
        reports.append((model_name + '_u_star', model, metrics))


    for model_name, model in models.items():
        X_train = train_df[['N', 'n', 'h', 'c', 'mean_n', 'std_n', 'alpha_hat', 'beta_hat', 'u_star_hat']]
        y_train = train_df['z']

        X_test = test_df[['N', 'n', 'h', 'c', 'mean_n', 'std_n', 'alpha_hat', 'beta_hat', 'u_star_hat']]
        y_test = test_df['z']

        model.fit(X_train, y_train)

        test_df['predicted_u_star'] = model.predict(X_test) * X_test['u_star_hat']

        actual_cost_col = get_actual_cost_col(test_df)

        metrics = {'mean_cost': actual_cost_col.mean(),
                'median_cost': actual_cost_col.median(),
                'esti_time_dif_percent': mean_absolute_error_percent(test_df['u'], test_df['predicted_u_star']),
                'train_score': model.score(X_train, y_train),
                'test_score': model.score(X_test, y_test)}
        
        reports.append((model_name + '_z', model, metrics))


    mlflow.set_experiment(experiment_name)
    runs = mlflow.search_runs(experiment_names=[experiment_name])
    for run_id in runs.run_id:
        # Delete the model
        mlflow.delete_run(run_id)

    for model_name, model, metrics in reports:
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param('model_name', model_name)
            mlflow.log_metrics(metrics=metrics)
        

def main():
    generate_data.generate(TEST_CONFIG,
                            gen_test_file,
                            'app.log',
                            n=TOTAL)

    generate_data.generate(TRAINING_CONFIG,
                            gen_train_file,
                            'app.log',
                            n=TOTAL)

    experiment(gen_test_file=gen_test_file, gen_train_file=gen_train_file, experiment_name=EXPERIMENT)

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

        experiment(gen_test_file=n_gen_test_file, gen_train_file=n_gen_train_file, experiment_name=str(n) + EXPERIMENT)

if __name__ == '__main__':
    main()


