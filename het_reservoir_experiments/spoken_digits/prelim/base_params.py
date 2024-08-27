import NymphESN.nymphesn as nymph
import NymphESN.restrictedmatrix as rmatrix
import benchmarks.spoken_digits as digits
import numpy as np

TRuns = 10
data =digits.create_data(["/home/cw1647/phd/benchmarks/spoken_digits/ti46/AllTestData/", "/home/cw1647/phd/benchmarks/spoken_digits/ti46/AllTrainData"])

# finding some "sane" values for the base ESN experiments

def single_ESN(density):
    vhat_digits = data["test digits"]
    WERs = []
    for i in range(TRuns):
        print(i)
        test_esn = nymph.NymphESN(3, 64*3, 10, seed=i, density=density, svd_dv=1)
        results_train, results_test = digits.run_spoken_digits(test_esn, data)
        v_digits = digits.results_to_digits(results_test, data["test coords"])
        WER = digits.word_error_rate(v_digits, vhat_digits)
        WERs.append(float(WER))

    return sum(WERs)/len(WERs)

def scan_ESNs():
    density = 0.1
    densities = dict()
    while density < 1:
        print(density)
        nrmse = single_ESN(density)
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
        nrmse = single_ESN(density)
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

def single_so(density, so):
    print(so)
    vhat_digits = data["test digits"]
    WERs = []
    for i in range(TRuns):
        print(i)
        test_esn = nymph.NymphESN(3, 64*3, 10, seed=i, density=density, svd_dv=1)
        W = rmatrix.create_restricted_esn_weights(64*3, 64, 3, density, so)
        results_train, results_test = digits.run_spoken_digits(test_esn, data)
        v_digits = digits.results_to_digits(results_test, data["test coords"])
        WER = digits.word_error_rate(v_digits, vhat_digits)
        WERs.append(float(WER))
    return sum(WERs)/len(WERs)

def scan_so(N, n_substates, density):
    so_max = density/4
    so = (n_substates/N)**2
    inc = 0.025
    results = dict()
    while so < so_max:
        print("first go")
        results[so] = single_so(density, so)
        so += inc
    vals = np.array(list(results.values()))
    min = np.min(vals)
    print(f"{min=}")
    min_keys = [float(k) for k, v in results.items() if v == min]
    if len(min_keys) < 2:
        min_keys = np.array(min_keys)
        window_min = np.min(min_keys)
        window_max = window_min+0.1
    min_keys = np.array(min_keys)
    window_max = np.max(min_keys)
    window_min = np.min(min_keys)
    so = window_min
    print(f"{so=}")
    inc = 0.0025
    results_granular = dict()
    while so <= window_max:
        results_granular[so] = single_so(density, so)
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


print(scan_so(64*3, 3, 0.2))

