import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy.stats import gamma



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


def cal_actual_time(n, intervals):
    return sum(intervals[n:])

def cal_cost(c, h, actual_time, predicted_time):
    t_diff = actual_time - predicted_time
    if(t_diff > 0):
        return t_diff * h
    else:
        return c
