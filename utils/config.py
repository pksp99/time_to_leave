# Configuration for data generation

TRAINING_CONFIG = {
    'alpha_range': [2, 3, 4, 6],
    'beta_range': [1, 2, 3, 4],
    'h_range': [1 / 15, 2 / 15, 10 / 15],
    'c_range': [20, 25, 30, 10],
    'N_range': range(10, 40),
    'n_range': [5]
}

TEST_CONFIG = {
    'alpha_range': [2, 3, 5, 7],
    'beta_range': [1, 1.5, 2],
    'h_range': [1 / 20, 2 / 20, 10 / 25],
    'c_range': [15, 20, 25, 30],
    'N_range': range(10, 50),
    'n_range': [5]
}
