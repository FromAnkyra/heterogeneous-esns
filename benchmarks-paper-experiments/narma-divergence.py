import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def get_u(T, seed=None):
    # random stream of inputs u in range 0,0.5 in col 0, 1s (bias) in col 1
    if seed:
        np.random.seed(seed)
    return np.random.uniform(0.0, 0.5, T)


def narmafun(y, u, alpha, beta, gamma, delta):
    # y =[y(t-N+1, ..., y(t-1), y(t))], similar for u
    return alpha * y[-1] + beta * y[-1] * sum(y) + gamma * u[0] * u[-1] + delta


def run_narma(T, runs, lw):
    narmaparams = {
        "fn-5": [5, (0.3, 0.05, 1.5, 0.1)],
        "us-5": [5, (0.3, 0.05, 1.5, 0.2)],  
        "us-10": [10, (0.3, 0.05, 1.5, 0.1)],
        "fn-15": [15, (0.3, 0.05, 1.5, 0.1)],
        "rt-20": [20, (0.3, 0.05, 1.5, 0.01)],
        "fn-20": [20, (0.3, 0.05, 1.5, 0.1)],
        "us-20": [20, (0.25, 0.05, 1.5, 0.01)],
        "us-30": [30, (0.2, 0.04, 1.5, 0.001)],
        "dale-30": [30, (0.2, 0.004, 1.5, 0.001)]
    }
    narma_divergences= {}
        # initial NARMA values of y
    for param in narmaparams.keys():
        divergences = 0
        NARMA=narmaparams[param][0]
        for i in range(runs):
            u = get_u(T)
            for t in range(0, NARMA):
                y = [0] * NARMA
            for t in range(NARMA - 1, T - 1):
                y_Nt = [y[i] for i in range(t-NARMA+1, t)]
                u_Nt = [u[i] for i in range(t-NARMA+1, t)]
                y_t1 = narmafun(y_Nt, u_Nt, *narmaparams[param][1])  # y(t+1) = f(y(t), u(t), ...)
                y.append(y_t1)
                if y_t1>5:
                    divergences+=1
                    break
        narma_divergences[param] = divergences
    return narma_divergences

print(1000)
print(run_narma(1000, 100, 1))
print(10000)
print(run_narma(10000, 100, 1))

