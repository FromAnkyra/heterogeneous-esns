import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

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


def run_previous_output(system):
    vtarget = system['vtarget'].tolist()
    output = [0] + vtarget[:-1]
    system['v_last_result'] = output
    return

def run_average_last_two(system):
    vtarget = system['vtarget'].tolist()
    output = [0, 0] + [vtarget[i+1] + (vtarget[i+1]-vtarget[i])/2 for i in range(len(vtarget)-2)]
    system['v_last_two'] = output
    return

def get_error(col_name, system):
    v = system[col_name].to_numpy()
    vhat = system['vtarget'].to_numpy()
    N = len(v)
    sumsq = sum((vhat - v)**2)
    vhatmean = sum(vhat) / N
    vhatminusvhatmeansq = sum((vhat - np.asarray([vhatmean] * N))**2)
    res = math.sqrt(sumsq / vhatminusvhatmeansq)
    return res

# ===================================================================
# ===================================================================
# reservoir parameters
NARMA = 10  # which NARMA function to use
Ttot = 1000

Nruns = 100  # number of train/test runs per exptl configuration

# config values
# Fig5a -- flat narma
# Fig5b -- naive physical narma
# Fig5c -- full physical narma with shifted output

col_names = ["last result", "average last two"]
errordf = pd.DataFrame(columns=col_names)

for seed in range(Nruns):
    np.random.seed(seed)

    system = pd.DataFrame(columns=['u', 'y','v_last_result', 'v_last_two', 'vtarget'])  # holds all the time states
    # get the random input stream
    system['u'] = get_u(Ttot)
    # get the NARMA states and outputs
    run_narma(NARMA, Ttot, system)
    get_narma_output(system, False)
    run_previous_output(system)
    run_average_last_two(system)
    errordf.at[seed, "last result"] = get_error('v_last_result', system)
    errordf.at[seed, "average last two"] = get_error('v_last_two', system)

last_result = errordf["last result"].to_numpy()
last_two = errordf["average last two"].to_numpy()

av_last_result = last_result.mean()
sd_last_result = last_result.std()

av_last_two = last_two.mean()
sd_last_two = last_two.std()

print(f"{av_last_result=}\n{sd_last_result=}\n{av_last_two=}\n{sd_last_two=}")
