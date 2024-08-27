import numpy as np
import NymphESN.nymphesn as nymph
import NymphESN.errorfuncs
import NymphESN.restrictedmatrix as rmatrix
import tempESN.TempESN as temp
import benchmarks.mso as mso
import pandas as pd
import sys

resolution_4 = np.linspace(0, 500, 4000)
MSO_eight = mso.generate_MSO(resolution_4, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value, mso.MSO.four.value, mso.MSO.five.value, mso.MSO.six.value, mso.MSO.seven.value, mso.MSO.eight.value])
MSO_eight = MSO_eight/16 # scale the input to be between [-0.5, 0.5]


rho = 2
density = 0.1
subreservoir_N = 32
N = 256
rN = 50
NR = 8

TWashout = 100
TTest = 200
TTrain = 400

data_lengths = (TWashout, TTest, TTrain)

TRuns = int(sys.argv[1])

def single_density(density):
    full_train = []
    full_test = []
    TTot = TWashout + TTrain + TTest
    for i in range(TRuns):
        f = np.tanh
        ESN = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        train, test = mso.run_MSO_rr(ESN, MSO_eight, data_lengths)
        full_train.append(train)
        full_test.append(test)
    full_train = np.array(full_train)
    full_test = np.array(full_test)
    train_median = np.median(full_train) # this should probably get discarded - TODO: discuss with S and M
    test_median = np.median(full_test)
    return test_median

def scan_densities():
    density = 0.1
    densities = dict()
    while density < 1:
        nrmse = single_density(density)
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
        nrmse = single_density(density)
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
        inner_size = N//n_substates
        f = np.tanh
        ESN = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        W = rmatrix.create_restricted_esn_weights(N, inner_size, n_substates, within, outwith) 
        ESN.set_weights(W)
        train, test = mso.run_MSO_rr(ESN, MSO_eight, data_lengths)
        full_train.append(train)
        full_test.append(test)
    full_train = np.array(full_train)
    full_test = np.array(full_test)
    train_median = np.median(full_train) # this should probably get discarded - TODO: discuss with S and M
    test_median = np.median(full_test)

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
    density = scan_densities()
    f = np.tanh
    p = scan_p(N, n_subgroups, density)
    within, outwith = find_distribution(n_subgroups, p, density)
    within = min(within, 1)
    outwith = max(outwith, 0)
    inner_size = N//n_subgroups
    TTot = TWashout + TTrain + TTest
    nrmsedf = pd.DataFrame(columns=['strain', 'stest', 'rtrain', 'rtest'])
    msedf = pd.DataFrame(columns=['strain', 'stest', 'rtrain', 'rtest'])
    rhythm_two = np.array([[1],[0, 1]], dtype=object)
    for i in range(TRuns):
        W = rmatrix.create_restricted_esn_weights(N, inner_size, n_subgroups, within, outwith)
        standard = nymph.NymphESN(1, N, 1, density=density, seed=i, svd_dv=0.1)
        restricted = nymph.NymphESN(1, N, 1, density=density, seed=i, svd_dv=0.1)
        restricted.set_weights(W)
        snrmse, smse = mso.run_MSO_rr(standard, MSO_eight, data_lengths, error="both")
        rnrmse, rmse = mso.run_MSO_rr(restricted, MSO_eight, data_lengths, error="both")
        
        #nrmse test error
        nrmsedf.at[i, 'stest'] = snrmse[1]
        nrmsedf.at[i, 'rtest'] = rnrmse[1]
    
    # save nrmses
    buf_nrmse = f"/home/cw1647/phd/UCNC-experiments/results/UCNC_extension/mso_eight_overall/size{N}-subgroups{n_subgroups}-density-{density}-p-{p}"
    nrmsedf.to_csv(buf_nrmse)

    # save mses 

    # compare
    
print("MSO eight fair")

for size in [512]:
    for item in [2, 4, 8]:
        print(f"N={size}, n_subgroups={item}")
        compare_distributions(N=size, n_subgroups=item)