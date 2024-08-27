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
        print(f"{len(y) - T=}")
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

def find_pmax(N, n_substates, density):
    pmax = min(((N*N*density)/n_substates) - n_substates + 1, (density*n_substates - 1)/(n_substates - 1))
    if pmax < 0:
        pmax = ((N*N*density)/n_substates) - n_substates + 1
    if pmax < 1:
        pmax = 1/pmax
    return pmax

def find_distribution(n_substates, p, density):
    within = (p*density*n_substates)/(p+n_substates-1)
    outwith = within/p
    if within < 0 or within > 1 or outwith < 0 or outwith > 1:
        print(f"{within=}, {outwith=}, {p=}, {density=}")
    return within, outwith

def single_p(N, n_substates, p, density):
    within, outwith = find_distribution(n_substates, p, density)
    within = min(within, 1)
    outwith = max(outwith, 0)
    full_train = []
    full_test = []
    TTot = TWashout + TTrain + TTest
    for i in range(TRuns):
        f = np.tanh
        inner_size = N//n_substates
        ESN = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        W = rmatrix.create_restricted_esn_weights(N, inner_size, n_substates, within, outwith) 
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


def scan_p(N, n_substates, density):
    pmax = find_pmax(N, n_substates, density)
    p = 1
    inc = (pmax-p)/10
    results = dict()
    if p == pmax or p > pmax: 
        print(p, pmax)
    while p <= pmax:
        p += inc
        results[p] = single_p(N, n_substates, p, density) # something is weird and i think my math may be wrong: if the stuff is consistently worse, shouldn't this just always give p=1??
    vals = np.array(list(results.values()))
    min = np.min(vals)

    min_keys = [k for k, v in results.items() if v == min]
    if len(min_keys) < 2:
        index = np.argwhere(vals==min)
        vals = np.delete(vals, index)
        min = np.min(vals)
        min_keys += [k for k, v in results.items() if v == min]
    min_keys = np.array(min_keys)
    window_max = np.max(min_keys)
    window_min = np.min(min_keys)
    p = window_min
    inc = abs(window_min - window_max)/10
    results_granular = dict()
    while p <= window_max:
        results_granular[p] = single_p(N, n_substates, p, density)
        p += inc
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
    # density = scan_densities(N)
    density = 0.1
    f = np.tanh
    p = scan_p(N, n_subgroups, density)
    within, outwith = find_distribution(n_subgroups, p, density)
    within = min(within, 1)
    outwith = max(outwith, 0)
    inner_size = N//n_subgroups
    TTot = TWashout + TTrain + TTest
    errordf = pd.DataFrame(columns=['strain', 'stest', 'rtrain', 'rtest'])
    # compare
    for i in range(50):
        standard = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        restricted = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        W = rmatrix.create_restricted_esn_weights(N, inner_size, n_subgroups, within, outwith)
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
        # standard.train_reservoir(vtarget_np[TWashout:TWashout+TTrain])
        standard.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])
        restricted.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])
        standard.get_output()
        restricted.get_output()
        strain, stest = standard.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        rtrain, rtest = restricted.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        errordf.at[i, 'strain'] = strain
        errordf.at[i, 'stest'] = stest
        errordf.at[i, 'rtrain'] = rtrain
        errordf.at[i, 'rtest'] = rtest
    buf = f"/home/cw1647/phd/UCNC-experiments/results/UCNC_extension/narma_overall/size-{N}-subgroups-{n_subgroups}-density-{density}-p-{p}.csv"
    # bufvis = f"/home/cw1647/phd/UCNC-experiments/results/narma-fair-again/size-{N}-subgroups-{n_subgroups}-density-{density}-p-{p}.svg"
    errordf.to_csv(buf)
    # vis.ErrorVis.vis(errordf, bufvis)
    #save that to a file somewhere
    return

print("narma fair")

for size in [64, 128, 256, 512]:
    for substates in [2, 4, 8]:
        print(size, substates)
        compare_distributions(size, substates)




