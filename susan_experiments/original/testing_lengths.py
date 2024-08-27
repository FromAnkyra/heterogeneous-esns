# train reservoirs on NARMA benchmark
import numpy as np
import pandas as pd

import reservoir


def get_u(T, seed=None):
    # random stream of inputs u in range 0,0.5 in col 0, 1s (bias) in col 1
    if seed:
        np.random.seed(seed)
    return np.random.uniform(0.0, 0.5, T)


def narmafun(y, u, alpha, beta, gamma, delta):
    # y =[y(t-N+1, ..., y(t-1), y(t))], similar for u
    return alpha * y[-1] + beta * y[-1] * sum(y) + gamma * u[0] * u[-1] + delta


def run_narma(NARMA, T, system, debug=False):
    narmaparams = {
        3: (0.3, 0.05, 1.5, 0.1),  # for testing only
        10: (0.3, 0.05, 1.5, 0.1),
        20: (0.25, 0.05, 1.5, 0.01),
        30: (0.2, 0.04, 1.5, 0.001)
    }

    # initial NARMA values of y
    for t in range(0, NARMA):
        system.at[t, 'y'] = 0

    for t in range(NARMA - 1, T - 1):
        lastN = system.loc[t - NARMA + 1:t]
        y_Nt = lastN['y'].tolist()
        u_Nt = lastN['u'].tolist()
        y_t1 = narmafun(y_Nt, u_Nt, *narmaparams[NARMA])  # y(t+1) = f(y(t), u(t), ...)
        system.at[t + 1, 'y'] = y_t1
    if(debug):
        print('u =', system['u'].tolist())
        print('y =', system['y'].tolist())


def get_narma_output(system, shift):
    vtarget = system['y'].tolist()
    if shift:
        # shift forward by one timestep: vtarget(t+1) = y(t)
        vtarget = [0] + vtarget[:-1]
    system['vtarget'] = vtarget


# ===================================================================
# ===================================================================
# reservoir parameters
rho = 2
density = 0.2
N = 50  # no of nodes
NARMA = 10  # which NARMA function to use

Twashout = 100
Ttrain = 1000
Ttest = 500
Ttot = Twashout + Ttrain + Ttest

Nruns = 200  # number of train/test runs per exptl configuration
config = [50, 200, 500]
config_name = ['50', '200', '500']

# config values
# Fig5a -- flat narma
# Fig5b -- naive physical narma
# Fig5c -- full physical narma with shifted output

col_names = [['train ' + str(conf), 'test ' + str(conf)] for conf in config_name]
col_names = [item for sublist in col_names for item in sublist]
errordf = pd.DataFrame(columns=col_names)

for seed in range(Nruns):
    # print(seed)
    Ttot = Twashout + Ttrain + Ttest
    np.random.seed(seed)
    system = pd.DataFrame(columns=['u', 'x', 'v', 'y', 'vtarget'])  # holds all the time states

    system['y'] = [np.inf] * Ttot    # to catch narma divergence
    while system.loc[Ttot - 1, 'y'] == np.inf:
        # get the random input stream
        system['u'] = get_u(Ttot)
        # get the NARMA states and outputs
        run_narma(NARMA, Ttot, system)

    # reservoir random weights
    rc = reservoir.Reservoir(N, rho=rho, density=density, seed=seed)

    for c, conf in enumerate(config):
        Ttest = conf
        rc.set_config('flat')
        if c == 0 or c == 1:
            shift_narma_output = False
        else:  # c==2
            shift_narma_output = True
        get_narma_output(system, shift_narma_output)

        # run the reservoir using previous/current time input
        rc.run_reservoir(Ttot, system)

        # train reservoir
        Wv = rc.train_reservoir(Twashout, Ttrain, system)

        # get training and testing errors
        rc.get_reservoir_output(Twashout + Ttrain + Ttest, Wv, system)
        train_str = 'train ' + str(config_name[c])
        test_str = 'test ' + str(config_name[c])
        errordf.at[seed, train_str] = rc.get_error(Twashout, Ttrain, system)
        errordf.at[seed, test_str] = rc.get_error(Twashout + Ttrain, Ttest, system)


reservoir.plot_results(system, errordf, Twashout, Ttrain, Ttest, 'narma.pdf', boxplot_only=True)
