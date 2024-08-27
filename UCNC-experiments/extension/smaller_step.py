import numpy as np
import NymphESN.nymphesn as nymph
import NymphESN.errorfuncs
import NymphESN.restrictedmatrix as rmatrix
import NymphESN.vis as vis
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

TRuns = 50

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

def scan_densities_small_step():
    density = 0.0025
    densities = dict()
    while density < 0.2:
        nrmse = single_density(density)
        print(f"{density=}, {nrmse=}")
        densities[float(density)] = float(nrmse)
        density += 0.0025
    print(f"{densities=}")
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

def single_So(N, n_substates, so, density):

    full_train = []
    full_test = []
    TTot = TWashout + TTrain + TTest
    for i in range(TRuns):
        f = np.tanh
        inner_size = N//n_substates
        ESN = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density, svd_dv=0.1)
        W = rmatrix.create_restricted_esn_weights(N, inner_size, n_substates, density, so) 
        ESN.set_weights(W)
        train, test = mso.run_MSO_rr(ESN, MSO_eight, data_lengths)
        full_train.append(train)
        full_test.append(test)
        full_train.append(train)
        full_test.append(test)

    full_train = np.array(full_train)
    full_test = np.array(full_test)
    train_median = np.median(full_train) # this should probably get discarded - TODO: discuss with S and M
    test_median = np.median(full_test)
    return test_median

def scan_so(N, n_substates, density):
    so_max = density/4
    so = 0
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
    # DW = scan_densities()
    D_small_step = scan_densities_small_step()
    f = np.tanh
    # DB = scan_so(N, n_subgroups, density)
    inner_size = N//n_subgroups
    TTot = TWashout + TTrain + TTest
    nrmsedf = pd.DataFrame(columns=['stest', 'sstest', 'rtest'])
    msedf = pd.DataFrame(columns=['strain', 'stest', 'rtrain', 'rtest', 'gtrain'])
    rhythm_two = np.array([[1],[0, 1]], dtype=object)
    for i in range(TRuns):
        # W = rmatrix.create_restricted_esn_weights(N, inner_size, n_subgroups, DW, DB)
        # standard = nymph.NymphESN(1, N, 1, density=DW, seed=i, svd_dv=0.1)
        small_step = nymph.NymphESN(1, N, 1, density=D_small_step, seed=i, svd_dv=0.1)
        # restricted = nymph.NymphESN(1, N, 1, density=DW, seed=i, svd_dv=0.1)
        # restricted.set_weights(W)
        # snrmse, smse = mso.run_MSO_rr(standard, MSO_eight, data_lengths, error="both")
        # rnrmse, rmse = mso.run_MSO_rr(restricted, MSO_eight, data_lengths, error="both")
        ssnrmse, ssmse = mso.run_MSO_rr(small_step, MSO_eight, data_lengths, error="both")
        #nrmse test error
        # nrmsedf.at[i, 'stest'] = snrmse[1]
        # nrmsedf.at[i, 'rtest'] = rnrmse[1]
        nrmsedf.at[i, 'sstest'] = ssnrmse[1]
    
    # save nrmses
    # buf_nrmse = f"/home/cw1647/phd/UCNC-experiments/sanity/small_step-{D_small_step}.csv"
    # nrmsedf.to_csv(buf_nrmse)
    # bufvis = f"/home/cw1647/phd/UCNC-experiments/sanity/small_step-{D_small_step}.png"
    # vis.ErrorVis.vis(nrmsedf, bufvis, low=0, high=0.2)


    # save mses 

    # compare
    
print("MSO eight physical")

for size in [512]:
    for item in [8]:
        print(f"N={size}, n_subgroups={item}")
        compare_distributions(N=size, n_subgroups=item)