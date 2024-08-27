from importlib_metadata import distribution
import NymphESN.nymphesn as nymph
import numpy as np
import NymphESN.errorfuncs as errorfunc
import NymphESN.restrictedmatrix as rmatrix
import NymphESN.vis as vis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_u(T, seed=None):
    # random stream of inputs u in range 0,0.5 in col 0, 1s (bias) in col 1
    if seed:
        np.random.seed(seed)
    return np.random.uniform(0.0, 0.5, T)


def narmafun(y, u, alpha, beta, gamma, delta):
    # y =[y(t-N+1, ..., y(t-1), y(t))], similar for u
    return alpha * y[-1] + beta * y[-1] * sum(y) + gamma * u[0] * u[-1] + delta


def run_narma(NARMA, T, u, debug=False):
    narmaparams = {
        5: (0.3, 0.05, 1.5, 0.2),  
        10: (0.3, 0.05, 1.5, 0.1),
        20: (0.25, 0.05, 1.5, 0.01),
        30: (0.2, 0.04, 1.5, 0.001)
    }

    # initial NARMA values of y
    for t in range(0, NARMA):
        y = [0] * NARMA

    for t in range(NARMA - 1, T - 1):
        
        y_Nt = [y[i] for i in range(t-NARMA+1, t)]
        u_Nt = [u[i] for i in range(t-NARMA+1, t)]
        y_t1 = narmafun(y_Nt, u_Nt, *narmaparams[NARMA])  # y(t+1) = f(y(t), u(t), ...)
        y.append(y_t1)
    if(debug):
        print('u =', u)
        print('y =', y)
    return [0] + y

rho = 2
density = 0.1
N = 16
rN = 50
NR = 2

TWashout = 100
TTest = 1000
TTrain = 1000

TRuns = 50

NARMA = 10

def vis(errordf, filename):
        sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})
        fig, ax = plt.subplots(1, 1, figsize=(2.5 * 4, 5))
        #my_pal = {'stest': 'gold', 'strain': 'darkorange', 'rtest': 'teal', 'rtrain': 'blue'}

        sns.boxplot(data=errordf, ax=ax, notch=True, width=0.6, linewidth=0.5, fliersize=0)
        ax.set_ylabel('NRMSE')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(0, 1.2)
        # plt.show()
        fig.savefig(filename, format="svg")

def single_density(density, N, errordf):
    full_train = []
    full_test = []
    TTot = TWashout + TTrain + TTest
    dens_string = f"{round(density, 3)}"
    for i in range(TRuns):
        f = np.tanh
        ESN = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        input = get_u(T=TTot, seed=i)
        vtarget = run_narma(NARMA, TTot, input, debug=False)
        # vtarget = np.hstack((np.array([0, 0, 0]), input[:-3]))
        vtarget_np = np.array(vtarget)

        ESN.set_data_lengths(TWashout, TTrain, TTest)
        ESN.set_input_stream(input)
        ESN.run_full()
        ESN.train_reservoir(vtarget_np[TWashout:TWashout+TTrain])
        ESN.get_output()
        train, test = ESN.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        full_train.append(train)
        full_test.append(test)

    try:
        errordf.insert(1, dens_string, full_test)
    except ValueError:
        x = 0
    full_train = np.array(full_train)
    full_test = np.array(full_test)
    train_median = np.median(full_train) # this should probably get discarded - TODO: discuss with S and M
    test_median = np.median(full_test)
    return test_median

def scan_densities(N): #TODO: make granularity a parametre
    density = 0.1
    densities = dict()
    errordf = pd.DataFrame(columns=["col0", "col1"])
    while density < 1:
        nrmse = single_density(density, N, errordf)
        densities[float(density)] = float(nrmse)
        density += 0.1
    vals = np.array(list(densities.values()))
    min = np.min(vals)

    min_keys = [float(k) for k, v in densities.items() if v == min]
    print(min_keys)
    if len(min_keys) < 2:
        min_keys = np.array(min_keys)
        window_min = np.min(min_keys)
        window_max = window_min+0.1
    else:
        min_keys = np.array(min_keys)
        window_max = np.max(min_keys)
        window_min = np.min(min_keys)
    density = window_min
    densities_granular = dict()
    while density <= window_max:
        nrmse = single_density(density, N, errordf)
        densities_granular[float(density)] = float(nrmse)
        density += 0.01
    vals = np.array(list(densities_granular.values()))
    min = np.min(vals)
    min_keys = [k for k, v in densities_granular.items() if v == min]
    if len(min_keys) > 1: #the unlikely chance we have the same NMSRE twice
        if abs(min_keys[0]-min_keys[1]) > 0.01: # the densities are not adjacent - two peaks
            min_keys = min_keys[0] # take a random one
        else:
            min_keys = min_keys[0]# take the midpoint b/w the two
    else: 
        min_keys = min_keys[0]
    errordf.pop("col0")
    errordf.pop("col1")
    buf = "/home/cw1647/phd/UCNC-experiments/results/density_error.csv"
    errordf.to_csv(buf)
    buf_vis = "/home/cw1647/phd/UCNC-experiments/results/density_error.svg"
    vis(errordf, buf_vis)
    return min_keys


scan_densities(128)