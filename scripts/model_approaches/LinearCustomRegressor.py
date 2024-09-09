import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sympy import symbols, lambdify
from utils import methods, math_expressions as me

class MLModel:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.train_cols = ['N', 'n', 'mean_n', 'std_n', 'alpha_hat', 'beta_hat', 'u_star_hat']
        self.target_col = 'u'
        self.preprocess_data()

    def preprocess_data(self):
        self.X_train = self.train_df[self.train_cols].values
        self.y_train = self.train_df[self.target_col].values
        self.X_test = self.test_df[self.train_cols].values
        self.y_test = self.test_df[self.target_col].values

        # Add bias term
        self.X_train = np.hstack((np.ones((self.X_train.shape[0], 1)), self.X_train))
        self.X_test = np.hstack((np.ones((self.X_test.shape[0], 1)), self.X_test))

    def objective_function(self, theta, X, y, h, c):
        n = len(X)
        predictions = X.dot(theta)
        errors = predictions - y
        cost = np.array([me.cus_cost_expr_1_eval(h_i, c_i, d_i) for h_i, c_i, d_i in zip(h, c, errors)])
        return np.sum(cost) / n

    def callback(self, xk):
        self.iteration += 1
        print(f"Iteration {self.iteration}: x = {xk}")
        print(me.get_POI_cus_cost_expr_1.cache_info())

    def train(self):
        initial_theta = np.zeros(self.X_train.shape[1])
        self.iteration = 0
        self.model = minimize(fun=self.objective_function, x0=initial_theta, args=(self.X_train, self.y_train, self.train_df['h'], self.train_df['c']), method='Nelder-Mead', options={'disp':True, 'return_all':True, 'maxiter':10000})

    def get_predictions(self):
        return self.X_test.dot(self.model.x)
    
    def evaluate_model(self):
        self.test_df['predicted_u_star'] = self.X_test.dot(self.model.x)
        self.test_df['actual_cost'] = self.test_df.apply(lambda row: methods.cal_cost(row['c'], row['h'], row['u'], row['predicted_u_star']), axis=1)
        actual_mean_cost = self.test_df['actual_cost'].mean()
        actual_median_cost = self.test_df['actual_cost'].median()
        print(f'Actual Mean cost: {actual_mean_cost:.2f}, Actual Median cost: {actual_median_cost:.2f}')
