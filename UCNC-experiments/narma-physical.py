'''
What do I need to do?

find “perfect” sparsities for sizes [16, 64, 128, 256] for the NARMA10 task with 3000 training steps
first do a coarse search: 0.1?
then do a finer search between the two highest points: 0.01 or 0.001?
once this is done, find some distribution that works using those sparsities for a two and four substate restricted reservoir
have written up how the sparsities are to be calculated given Stotal in this document https://www.overleaf.com/5167969355cmxcktsfjnfj
run the experiments on the three reservoirs for each task
'''
from importlib_metadata import distribution
import NymphESN.nymphesn as nymph
import numpy as np
import NymphESN.errorfuncs as errorfunc
import NymphESN.restrictedmatrix as rmatrix
import NymphESN.vis as vis
import pandas as pd
import sys

#single density
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
    return y


# def get_narma_output(shift):
#     vtarget = system['y'].tolist()
#     if shift:
#         # shift forward by one timestep: vtarget(t+1) = y(t)
#         vtarget = [0] + vtarget[:-1]
#     return vtarget

# =========================================================================================================
# =========================================================================================================

rho = 2
density = 0.1
N = 16
rN = 50
NR = 2

TWashout = 100
TTest = 1000
TTrain = 1000

TRuns = int(sys.argv[1])

NARMA = 10

def single_density(density, N):
    full_train = []
    full_test = []
    TTot = TWashout + TTrain + TTest
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
        ESN.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])
        ESN.get_output()
        train, test = ESN.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        full_train.append(train)
        full_test.append(test)

    full_train = np.array(full_train)
    full_test = np.array(full_test)
    train_median = np.median(full_train) # this should probably get discarded - TODO: discuss with S and M
    test_median = np.median(full_test)
    return test_median

def scan_densities(N): #TODO: make granularity a parametre
    density = 0.1
    densities = dict()
    while density < 1:
        nrmse = single_density(density, N)
        densities[float(density)] = float(nrmse)
        density += 0.1
    vals = np.array(list(densities.values()))
    min = np.min(vals)

    min_keys = [float(k) for k, v in densities.items() if v == min]
    if len(min_keys) < 2:
        min_keys = np.array(min_keys)
        window_min = np.min(min_keys)
        window_max = window_min+0.1
    min_keys = np.array(min_keys)
    window_max = np.max(min_keys)
    window_min = np.min(min_keys)
    density = window_min
    densities_granular = dict()
    while density <= window_max:
        nrmse = single_density(density, N)
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
    return min_keys

def single_So(N, n_substates, so, density):

    full_train = []
    full_test = []
    TTot = TWashout + TTrain + TTest
    for i in range(TRuns):
        f = np.tanh
        inner_size = N//n_substates
        ESN = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        W = rmatrix.create_restricted_esn_weights(N, inner_size, n_substates, density, so) 
        input = get_u(T=TTot, seed=i)
        vtarget = run_narma(NARMA, TTot, input, debug=False)
        # vtarget = np.hstack((np.array([0, 0, 0]), input[:-3]))
        vtarget_np = np.array(vtarget)
        ESN.set_data_lengths(TWashout, TTrain, TTest)
        ESN.set_input_stream(input)
        ESN.run_full(W = W)
        ESN.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])
        ESN.get_output()
        train, test = ESN.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        full_train.append(train)
        full_test.append(test)

    full_train = np.array(full_train)
    full_test = np.array(full_test)
    train_median = np.median(full_train) # this should probably get discarded - TODO: discuss with S and M
    test_median = np.median(full_test)
    return test_median


def scan_so(N, n_substates, density):
    so_max = density/4
    so = (n_substates/N)**2
    inc = 0.025
    results = dict()
    while so < so_max:
        results[so] = single_So(N, n_substates, so, density)
        so += inc
    vals = np.array(list(results.values()))
    min = np.min(vals)

    min_keys = [float(k) for k, v in results.items() if v == min]
    if len(min_keys) < 2:
        min_keys = np.array(min_keys)
        window_min = np.min(min_keys)
        window_max = window_min+0.1
    min_keys = np.array(min_keys)
    window_max = np.max(min_keys)
    window_min = np.min(min_keys)
    so = window_min
    inc = 0.0025
    results_granular = dict()
    while so <= window_max:
        results_granular[so] = single_So(N, n_substates, so, density)
        so += inc
    vals = np.array(list(results_granular.values()))
    min = np.min(vals)
    min_keys = [k for k, v in results_granular.items() if v == min]
    if len(min_keys) > 1: #the unlikely chance we have the same NMSRE twice
        if abs(min_keys[0]-min_keys[1]) > 0.01: # the densities are not adjacent - two peaks
            min_keys = min_keys[0] # take a random one
        else:
            min_keys = (min_keys[0] + min_keys[1])/2# take the midpoint b/w the two
    else:
        min_keys = min_keys[0]
    return min_keys

def compare_distributions(N, n_subgroups):
    density = scan_densities(N)
    f = np.tanh
    so = scan_so(N, n_subgroups, density)
    inner_size = N//n_subgroups
    TTot = TWashout + TTrain + TTest
    errordf = pd.DataFrame(columns=['strain', 'stest', 'rtrain', 'rtest'])
    # compare
    for i in range(TRuns):
        standard = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        restricted = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        W = rmatrix.create_restricted_esn_weights(N, inner_size, n_subgroups, density, so)
        input = get_u(T=TTot, seed=i)
        vtarget = run_narma(NARMA, TTot, input, debug=False)
        # vtarget = np.hstack((np.array([0, 0, 0]), input[:-3]))
        vtarget_np = np.array(vtarget)
        standard.set_data_lengths(TWashout, TTrain, TTest)
        restricted.set_data_lengths(TWashout, TTrain, TTest)
        standard.set_input_stream(input)
        restricted.set_input_stream(input)
        standard.run_full()
        restricted.run_full(W = [W])
        standard.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])
        restricted.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])
        standard.get_output()
        restricted.get_output()
        strain, stest = standard.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        rtrain, rtest = restricted.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        # errordf.at[i, 'strain'] = strain
        errordf.at[i, 'stest'] = stest
        # errordf.at[i, 'rtrain'] = rtrain
        errordf.at[i, 'rtest'] = rtest
    buf = f"/home/cw1647/phd/UCNC-experiments/results/UCNC_extension/narma_patch/size-{N}-subgroups-{n_subgroups}-sw-{density}-so-{so}.csv"
    # bufvis = f"/home/cw1647/phd/UCNC-experiments/results/narma-physical-again/size-{N}-subgroups-{n_subgroups}-sw-{density}-so-{so}.svg"
    # vis.ErrorVis.vis(errordf, bufvis)
    errordf.to_csv(buf)
    #save that to a file somewhere
    return

print("narma physical")
for item in [64, 128, 256, 512]:
    for subs in [2, 4, 8]:
        print(item, subs)
        compare_distributions(item, subs)




