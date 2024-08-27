import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
def get_u(T, seed=None):
    # random stream of inputs u in range 0,0.5 in col 0, 1s (bias) in col 1
    if seed:
        np.random.seed(seed)
    return np.random.uniform(0.0, 0.5, T)


def narmafun(y, u, alpha, beta, gamma, delta, tanh=False):
    # y =[y(t-N+1, ..., y(t-1), y(t))], similar for u
    if tanh:
        return np.tanh(alpha * y[-1] + beta * y[-1] * sum(y) + gamma * u[0] * u[-1] + delta)
    return alpha * y[-1] + beta * y[-1] * sum(y) + gamma * u[0] * u[-1] + delta


def run_narma(T, u, lw):
    narmaparams = {
        5: (0.3, 0.05, 1.5, 0.1, False),  
        10: (0.3, 0.05, 1.5, 0.1, False),
        20: (0.3, 0.05, 1.5, 0.01, True),
        30: (0.2, 0.04, 1.5, 0.001, False)
    }
    narma_results = {}
        # initial NARMA values of y
    for NARMA in [5, 10, 20, 30]:
        for t in range(0, NARMA):
            y = [0] * NARMA
        for t in range(NARMA - 1, T - 1):
            y_Nt = [y[i] for i in range(t-NARMA+1, t)]
            u_Nt = [u[i] for i in range(t-NARMA+1, t)]
            y_t1 = narmafun(y_Nt, u_Nt, *narmaparams[NARMA])  # y(t+1) = f(y(t), u(t), ...)
            y.append(y_t1)
        
        font = {'family' : 'normal',
        'size'   : 18}

        matplotlib.rc('font', **font)    
        plt.plot(y[30:], label=f"NARMA {NARMA}", lw=lw)
        plt.legend()
        narma_results[NARMA] = list(y)
    return narma_results
df = pd.DataFrame(columns=['u', '5', '10', '20', '30'])
u = get_u(230)
narmas = run_narma(230, u, 1)
plt.show()

