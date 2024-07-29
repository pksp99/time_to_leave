from sympy import symbols, sympify, E, integrate, gamma, oo, lambdify, Eq, solve
import numpy as np
import plotly.graph_objs as go

def lower_incomplete_gamma(s, x ='x'):
    x = sympify(x)
    s = sympify(s)
    t = symbols('t')
    expr = (t ** (s - 1)) * (E ** (-t))
    return  integrate(expr, (t, 0, x))


def gamma_pdf(alpha, beta, x='x'):
    alpha = sympify(alpha)
    beta = sympify(beta)
    x = sympify(x)
    num = (x ** (alpha - 1)) * (E ** (-x/beta))
    den = gamma(alpha) * (beta ** alpha)
    return num / den

def gamma_cdf(alpha, beta, x='x'):
    alpha = sympify(alpha)
    beta = sympify(beta)
    x = sympify(x)
    return lower_incomplete_gamma(alpha, x / beta) / gamma(alpha)


def gamma_hazard_rate(alpha, beta, x='x'):
    alpha = sympify(alpha)
    beta = sympify(beta)
    return gamma_pdf(alpha, beta) / (1 - gamma_cdf(alpha, beta))

def gamma_cost(alpha, beta, h, c, x='x'):
    alpha = sympify(alpha)
    beta = sympify(beta)
    h = sympify(h)
    c = sympify(c)
    x = sympify(x)
    t = sympify('t')
    expr = integrate(t * gamma_pdf(alpha, beta, x=t), (t, x, oo))
    print(expr)
    return c * gamma_cdf(alpha, beta) - x * h * (1 - gamma_cdf(alpha, beta)) + h * expr


def plot_expression(expression, x_range, title, x_label, y_label, x='x'):
    x = sympify(x)
    expr_func = lambdify(x, expression, 'numpy')
    x_values = np.linspace(x_range[0], x_range[1], 400)
    y_values = expr_func(x_values)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=title))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    
    fig.show()
