def avg_model(N, n, intervals):
    observe = intervals[:n]
    avg_time = sum(observe) / n
    return avg_time * (N - n)