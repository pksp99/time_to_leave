import numpy as np
import plotly.graph_objs as go
from sympy import sympify, E, gamma, lowergamma, uppergamma, simplify, exp, Max, Eq, nsolve, S, Piecewise, re
from functools import lru_cache
import math


def gamma_pdf(alpha, beta, x='x'):
    alpha = sympify(alpha)
    beta = sympify(beta)
    x = sympify(x)
    num = (x ** (alpha - 1)) * (E ** (-x / beta))
    den = gamma(alpha) * (beta ** alpha)
    return num / den


def gamma_cdf(alpha, beta, x='x'):
    alpha = sympify(alpha)
    beta = sympify(beta)
    x = sympify(x)
    return lowergamma(alpha, x / beta) / gamma(alpha)


def gamma_hazard_rate(alpha, beta, x='x'):
    alpha = sympify(alpha)
    beta = sympify(beta)
    x = sympify(x)
    return gamma_pdf(alpha, beta, x=x) / (1 - gamma_cdf(alpha, beta, x=x))


def gamma_cost(alpha, beta, h, c, x='x'):
    alpha = sympify(alpha)
    beta = sympify(beta)
    h = sympify(h)
    c = sympify(c)
    x = sympify(x)

    # Simplified the integral for performance efficiency
    expr = beta * uppergamma(alpha + 1, x / beta) / gamma(alpha)
    return c * gamma_cdf(alpha, beta, x=x) - x * h * (1 - gamma_cdf(alpha, beta, x=x)) + h * expr


def plot_expression(expression, x_range, title, x_label, y_label, x='x'):
    x = sympify(x)
    expression = simplify(expression)
    x_values = np.linspace(x_range[0], x_range[1], 400)

    # expr_func = lambdify(x, expression, 'numpy')
    # y_values = expr_func(x_values)

    y_values = [float(expression.subs(x, x_val).evalf()) for x_val in x_values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=title))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)

    fig.show()

def cus_cost_expr_1(h, c, d):
    # d => u_predict - u_actual => (x - u)
    d = sympify(d)
    h = sympify(h)
    c = sympify(c)
    f = c * (1 / (1 + exp(-c * (d))))
    g = -h * (d)
    sols = point_of_intersection(f, g, d)
    expr = Piecewise((g, d <= sols), (f, True))
    return expr

def cus_cost_expr_1_eval(h, c, d):
    poi = get_POI_cus_cost_expr_1(h, c)
    if d <= poi:
        return -h * d
    else:
        return c * (1 / (1 + math.e ** (-c * d)))

def cus_cost_expr_2_eval(h, c, d):
    poi = get_POI_cus_cost_expr_2(h, c)
    if d <= poi:
        return h * (-d)**(5/4)
    else:
        return h * (d)**(4)

def cus_cost_expr_2(h, c, d):
    # d => u_predict - u_actual => (x - u)
    d = sympify(d)
    h = sympify(h)
    c = sympify(c)
    g = h * (-d)**(5/4)
    f = h * (d)**(4)
    sols = point_of_intersection(f, g, d)
    expr = Piecewise((g, d <= sols), (f, True))
    return expr


def point_of_intersection(f, g, x):
    equation = Eq(f, g)
    sols = nsolve(equation, x, 0)
    return re(sols)

@lru_cache(maxsize=256)
def get_POI_cus_cost_expr_1(h, c):
    expr = cus_cost_expr_1(h, c, 'd')
    return float(expr.args[0][1].rhs)


@lru_cache(maxsize=256)
def get_POI_cus_cost_expr_2(h, c):
    expr = cus_cost_expr_2(h, c, 'd')
    return float(expr.args[0][1].rhs)
