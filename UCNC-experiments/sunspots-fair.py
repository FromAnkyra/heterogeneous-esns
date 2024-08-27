from importlib_metadata import distribution
import NymphESN.nymphesn as nymph
import numpy as np
import NymphESN.errorfuncs as errorfunc
import NymphESN.restrictedmatrix as rmatrix
import pandas as pd
import NymphESN.vis as vis
import sys

TWashout = 320
TTrain = 500
TTest = 2000
TRuns = int(sys.argv[1])

df = pd.read_csv('/home/cw1647/phd/benchmarks/monthly-sunspots.csv')
data = df['Sunspots'].to_numpy()
    # normalise data to range [0,0.5]
max = np.amax(data)
data = data / (2 * max)
sunspots = data.tolist()
input = np.array(sunspots)
vtarget = np.array(sunspots[1:]+[sunspots[-1]])
print(f"{vtarget.size}")

def run_sunspots(esn: nymph.NymphESN):
    train = train_sunspots(esn)
    test = test_sunspots(esn)
    return train, test
    
def train_sunspots(esn: nymph.NymphESN):
    esn.set_data_lengths(TWashout, TTrain, 0)
    esn.set_input_stream(sunspots[:TWashout+TTrain])
    esn.run_full()
    esn.train_ridge_regression(sunspots[TWashout:TWashout+TTrain])
    esn.get_output()
    train, _ = esn.get_error(sunspots[:TWashout+TTrain], errorfunc.ErrorFuncs.nrmse)
    return train

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
    full_train = []
    full_test = []
    for i in range(10):
        f = np.tanh
        ESN = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density, svd_dv=1.5)
        ESN.set_data_lengths(TWashout, TTrain, TTest)
        ESN.set_input_stream(input)
        ESN.run_full()
        ESN.train_reservoir(vtarget[TWashout:TWashout+TTrain])
        ESN.get_output()
        train, test = ESN.get_error(vtarget, errorfunc.ErrorFuncs.nrmse)
        full_train.append(train)
        full_test.append(test)
    full_train = np.array(full_train)
    full_test = np.array(full_test)
    train_median = np.median(full_train)
    test_median = np.median(full_test)
    return test_median 

def scan_densities(N, n_substates):
    density = 0.1
    densities = dict()
    while density < 1:
        p = find_pmax(N, n_substates, density)
        if p < 2:
            print(f"pog, {density=}, {p=}")
            break
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
    if pmax <= 0:
        pmax = ((N*N*density)/n_substates) - n_substates + 1
    if pmax < 1:
        pmax = 1/pmax
    return pmax

def find_distribution(n_substates, p, density):
    within = (p*density*n_substates)/(p+n_substates-1)
    outwith = within/p
    if within < 0 or within > 1 or outwith < 0 or outwith > 1:
        print(f"{within=}, {outwith=}, {p=}, {density=}")
        if within > 1:
            within = 1
    return within, outwith


def single_p(N, n_substates, p, density):
    within, outwith = find_distribution(n_substates, p, density)
    within = np.min([within, 1])
    outwith = np.max([outwith, 0])
    full_train = []
    full_test = []
    for i in range(10):
        f = np.tanh
        inner_size = N//n_substates
        ESN = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        W = rmatrix.create_restricted_esn_weights(N, inner_size, n_substates, within, outwith, svd_dv=1.5)
        ESN.set_data_lengths(TWashout, TTrain, TTest)
        ESN.set_input_stream(input)
        ESN.run_full(W=[W])
        ESN.train_reservoir(vtarget[TWashout:TWashout+TTrain])
        ESN.get_output()
        train, test = ESN.get_error(vtarget, errorfunc.ErrorFuncs.nrmse)
        full_train.append(train)
        full_test.append(test)
    full_train = np.array(full_train)
    full_test = np.array(full_test)
    train_median = np.median(full_train)
    test_median = np.median(full_test)
    return test_median 

def scan_p(N, n_substates, density):
    pmax = find_pmax(N, n_substates, density)
    p = 4
    inc = (pmax-p)/10
    results = dict()
    if p == pmax or p > pmax: 
        print(p, pmax)
        return pmax
    while p <= pmax:
        results[p] = single_p(N, n_substates, p, density) # something is weird and i think my math may be wrong: if the stuff is consistently worse, shouldn't this just always give p=1??
        p += inc
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

def compare_esns(N, n_subgroups):
    density = scan_densities(N, n_subgroups)
    f = np.tanh
    p = scan_p(N, n_subgroups, density)
    within, outwith = find_distribution(n_subgroups, p, density)
    within = np.min([within, 1])
    outwith = np.max([outwith, 0])
    inner_size = N//n_subgroups
    TTot = TWashout + TTrain + TTest
    errordf = pd.DataFrame(columns=['strain', 'stest', 'rtrain', 'rtest'])
    # compare
    for i in range(TRuns):
        standard = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        restricted = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        W = rmatrix.create_restricted_esn_weights(N, inner_size, n_subgroups, within, outwith, svd_dv=1.5)
        standard.set_data_lengths(TWashout, TTrain, TTest)
        restricted.set_data_lengths(TWashout, TTrain, TTest)
        standard.set_input_stream(input)
        restricted.set_input_stream(input)
        standard.run_full()
        restricted.run_full(W = [W])
        standard.train_reservoir(vtarget[TWashout:TWashout+TTrain])
        restricted.train_reservoir(vtarget[TWashout:TWashout+TTrain])
        standard.get_output()
        restricted.get_output()
        strain, stest = standard.get_error(vtarget, errorfunc.ErrorFuncs.nrmse)
        rtrain, rtest = restricted.get_error(vtarget, errorfunc.ErrorFuncs.nrmse)
        # errordf.at[i, 'strain'] = strain
        errordf.at[i, 'stest'] = stest
        # errordf.at[i, 'rtrain'] = rtrain
        errordf.at[i, 'rtest'] = rtest
    bufcsv = f"/home/cw1647/phd/UCNC-experiments/results/UCNC_extension/sunspots_overall/size-{N}-subgroups-{n_subgroups}-density-{density}-p-{p}.csv"
    # bufvis = f"/home/cw1647/phd/UCNC-experiments/results/sunspots-fair-again/size-{N}-subgroups-{n_subgroups}-density-{density}-p-{p}.svg"
    errordf.to_csv(bufcsv)
    # vis.ErrorVis.vis(errordf, bufvis)
    return
print("sunspots fair")
for size in [512]:
    for substates in [2, 4, 8]:
        print(size, substates)
        compare_esns(size, substates)