from importlib_metadata import distribution
import NymphESN.nymphesn as nymph
import numpy as np
import NymphESN.errorfuncs as errorfunc
import NymphESN.restrictedmatrix as rmatrix
import pandas as pd
import NymphESN.vis as vis
import sys

TWashout = 500
TTrain = 1500
TTest = 820
TRuns = int(sys.argv[1])

df = pd.read_csv('/home/cw1647/phd/benchmarks/monthly-sunspots.csv')
data = df['Sunspots'].to_numpy()
    # normalise data to range [0,0.5]
max = np.amax(data)
data = data / (2 * max)
sunspots = data.tolist()
input = np.array(sunspots)
vtarget = np.array(sunspots[1:]+[sunspots[-1]])

f = open("/home/cw1647/phd/benchmarks/monthly-sunspots.csv", "r")
lines = f.readlines()
lines = [line.split(",") for line in lines]
sunspots = np.array([float(line[1].strip("\n")) for line in lines[1:]])

def test_sunspots(esn: nymph.NymphESN):
    input = np.array(sunspots)
    outputs = []
    esn.set_data_lengths(TWashout, TTrain, TTest)
    for i in range(TTest):
        esn.set_input_stream(input)
        esn.run_timestep(i)
        esn.get_output()
        output = esn.vall[0][TWashout+TTrain+i]
        # print(output)

        # print(output)
        if i<TTest-1:
            input[0][TWashout+TTrain+i+1] = output
        outputs.append(output)
    inputs = np.array(sunspots[TWashout+TTrain:])
    outputs=np.array(outputs) # why is the max here not a numeric object??
    test = errorfunc.ErrorFuncs.nrmse(outputs, inputs)
    return test

def single_density(density, N):
    print("boop")
    full_train = []
    full_test = []
    for i in range(TRuns):
        print(i)
        f = np.tanh
        ESN = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        ESN = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        ESN.set_data_lengths(TWashout, TTrain, TTest)
        ESN.set_input_stream(input)
        ESN.run_full()
        ESN.train_ridge_regression(vtarget[TWashout:TWashout+TTrain])
        ESN.get_output()
        train, test = ESN.get_error(vtarget, errorfunc.ErrorFuncs.nrmse)
        full_train.append(train)
        full_test.append(test)
    full_train = np.array(full_train)
    full_test = np.array(full_test)
    train_median = np.median(full_train)
    test_median = np.median(full_test)
    return test_median 

def scan_densities(N):
    density = 0.1
    densities = dict()
    print("start density search")
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
    print("density found")
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
        ESN.set_weights(W)
        ESN.set_data_lengths(TWashout, TTrain, TTest)
        ESN.set_input_stream(input)
        ESN.run_full(W=[W])
        ESN.train_ridge_regression(vtarget[TWashout:TWashout+TTrain])
        ESN.get_output()
        train, test = ESN.get_error(vtarget, errorfunc.ErrorFuncs.nrmse)
        full_train.append(train)
        full_test.append(test)
    full_train = np.array(full_train)
    full_test = np.array(full_test)
    train_median = np.median(full_train)
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
    print("so found")
    return min_keys

def compare_esns(N, n_subgroups):
    density = scan_densities(N)
    f = np.tanh
    so = scan_so(N, n_subgroups, density)
    inner_size = N//n_subgroups
    TTot = TWashout + TTrain + TTest
    errordf = pd.DataFrame(columns=['strain', 'stest', 'rtrain', 'rtest'])
    # compare
    for i in range(TRuns):
        print(i)
        standard = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        restricted = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        W = rmatrix.create_restricted_esn_weights(N, inner_size, n_subgroups, density, so)
        standard.set_data_lengths(TWashout, TTrain, TTest)
        restricted.set_data_lengths(TWashout, TTrain, TTest)
        standard.set_input_stream(input)
        restricted.set_input_stream(input)
        standard.run_full()
        restricted.run_full(W = [W])
        standard.train_ridge_regression(vtarget[TWashout:TWashout+TTrain])
        restricted.train_ridge_regression(vtarget[TWashout:TWashout+TTrain])
        standard.get_output()
        restricted.get_output()
        strain, stest = standard.get_error(vtarget, errorfunc.ErrorFuncs.nrmse)
        rtrain, rtest = restricted.get_error(vtarget, errorfunc.ErrorFuncs.nrmse)
        # errordf.at[i, 'strain'] = strain
        errordf.at[i, 'stest'] = stest
        # errordf.at[i, 'rtrain'] = rtrain
        errordf.at[i, 'rtest'] = rtest
    bufcsv = f"/home/cw1647/phd/UCNC-experiments/results/UCNC_extension/sunspots_patch/size-{N}-n-subgroups-{n_subgroups}-sw-{density}-so-{so}.csv"
    # bufvis = f"/home/cw1647/phd/UCNC-experiments/results/sunspots-physical-again/size-{N}-subgroups-{n_subgroups}-sw-{density}-so-{so}.svg"
    errordf.to_csv(bufcsv)
    # vis.ErrorVis.vis(errordf, bufvis)
    return       
print("sunspots physical")
for size in [512]:
    for substates in [2, 4, 8]:
        print(size, substates)
        compare_esns(size, substates)