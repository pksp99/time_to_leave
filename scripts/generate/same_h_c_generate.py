from utils import math_expressions as generate_data



# Configuration for data generation
TRAINING_CONFIG = {
    'alpha_range': [2, 4, 6, 10],
    'beta_range': [round(i * 0.5, 1) for i in range(2, 9)],
    'h_range': [1 / 15],
    'c_range': [25],
    'N_range': range(10, 40),
    'n_range': [5]
}

TEST_CONFIG = {
    'alpha_range': [3, 5, 7, 8],
    'beta_range': [round(i * 0.5, 1) for i in range(2, 9)],
    'h_range': [1 / 15],
    'c_range': [25],
    'N_range': range(15, 40),
    'n_range': [5]
}

TOTAL = 40000

gen_test_file = 'gen_test_v3.csv'
gen_train_file = 'gen_train_v3.csv'

     

def main():
    generate_data.generate(TEST_CONFIG,
                            gen_test_file,
                            'app.log',
                            n=TOTAL)

    generate_data.generate(TRAINING_CONFIG,
                            gen_train_file,
                            'app.log',
                            n=TOTAL)