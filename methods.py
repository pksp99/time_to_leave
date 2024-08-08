import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy.stats import gamma
import statistics

import importlib

import math_expressions as mexpr
importlib.reload(mexpr)
from sympy import Eq, simplify, real_roots, solveset, S, symbols, solve, lambdify

from mpmath import mp

def plot_gamma(shape = 2, scale = 2):

    # Generate x values
    x_min = max(0, gamma.ppf(0.001, shape, scale=scale) - 1)
    x_max = gamma.ppf(0.999, shape, scale=scale) + 1
    x = np.linspace(x_min, x_max, 1000)

    # Calculate the gamma distribution probability density function (PDF) values
    y = gamma.pdf(x, shape, scale=scale)

    # Plot the gamma distribution
    trace = go.Scatter(x=x, y=y, mode='lines', name=f'Gamma Distribution (shape={shape}, scale={scale})')
    layout = go.Layout(title='Gamma Distribution',
                       xaxis=dict(title='x'),
                       yaxis=dict(title='Probability Density'),
                       template="plotly_white")

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

def plot_plotly(data, mode='lines', data_label='data'):

    mean_value = np.mean(data)
    median_value = np.median(data)

    # Create the plot
    fig = go.Figure()

    # Add trace for original data
    fig.add_trace(go.Scatter(y=data, mode=mode, name=f'{data_label} per iteration'))

    # Add trace for mean
    fig.add_trace(go.Scatter(x=[0, len(data)-1], y=[mean_value, mean_value],
                             mode='lines', name=f'Mean {mean_value:.2f}', line=dict(color='red', dash='dash')))

    # Add trace for median
    fig.add_trace(go.Scatter(x=[0, len(data)-1], y=[median_value, median_value],
                             mode='lines', name=f'Median {median_value:.2f}', line=dict(color='green', dash='dash')))

    # Update layout
    fig.update_layout(
        title=f'{data_label} over {len(data)} iterations',
        xaxis_title='Iteration',
        yaxis_title= data_label
    )

    # Show the plot
    fig.show()

def multi_plot_plotly(data:list, mode='lines', data_label=['data']):


    # Create the plot
    fig = go.Figure()

    for i in range(len(data)):
        fig.add_trace(go.Scatter(y=data[i], mode=mode, name=f'{data_label[i]} per iteration'))

    # Show the plot
    fig.show()

def cal_actual_time(n:int, intervals:list[float]) -> float:
    return sum(intervals[n:])

def cal_cost(c:float, h:float, actual_time:float, predicted_time:float) -> float:
    t_diff = actual_time - predicted_time
    if(t_diff > 0):
        return t_diff * h
    else:
        return c

def get_u_star(N:int, alpha:float, beta:float, h:float, c:float) -> float:
    expr = mexpr.gamma_hazard_rate(alpha * N, beta) - h / c
    s_expr = (simplify(expr * expr.as_numer_denom()[1]))
    u_star = real_roots(s_expr)
    u_star = [sol.evalf() for sol in u_star if sol.is_real and sol > 0]
    return float(u_star[0])

def gamma_estimate_parameters(n:int, intervals:list[float]) -> tuple[float, float]:
    mean = statistics.mean(intervals[:n])
    variance = statistics.variance(intervals[:n])
    beta = variance / mean
    alpha  = mean / beta
    return (alpha, beta)


def get_u_star_binary (N:int, alpha:float, beta:float, h:float, c:float, precision=8) -> float:

    x = symbols('x')
    f = lambda u: float(mexpr.gamma_hazard_rate(alpha * N, beta).subs(x, u).evalf(5))
    required_value = round(h/c, precision)
    step = 10 ** (-precision)

    # Find initial interval
    start, end = step, step * 2
    while f(end) < required_value:
        start, end = end, 2 * end

    # Perform binary search
    while start < end:
        mid = round((start + end) / 2, precision)
        y = round(f(mid), precision)
        if y == required_value:
            break
        elif y > required_value:
            end = mid - step
        else:
            start = mid + step

    return round((start + end) / 2, precision)


def get_u_star_binary_fast (N:int, alpha:float, beta:float, h:float, c:float, precision=8) -> float:

    f = lambda x: gamma.pdf(x, N * alpha, scale=beta)
    F = lambda x: gamma.cdf(x, N * alpha, scale=beta)
    ha = lambda x: f(x) / (1 - F(x))

    required_value = round(h/c, precision)
    step = 10 ** (-precision)

    # Find initial interval
    start, end = step, step * 2
    while ha(end) < required_value:
        start, end = end, 2 * end

    # Perform binary search
    while start < end:
        mid = round((start + end) / 2, precision)
        y = round(ha(mid), precision)
        if y == required_value:
            break
        elif y > required_value:
            end = mid - step
        else:
            start = mid + step

    return round((start + end) / 2, precision)
