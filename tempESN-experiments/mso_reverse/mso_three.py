import numpy as np
import NymphESN.nymphesn as nymph
import NymphESN.errorfuncs
import NymphESN.restrictedmatrix as rmatrix
import tempESN.TempESN as temp
import benchmarks.mso as mso
import pandas as pd
import sys

resolution_4 = np.linspace(0, 500, 4000)
MSO_three = mso.generate_MSO(resolution_4, [mso.MSO.eight.value, mso.MSO.seven.value, mso.MSO.six.value])
MSO_three = MSO_three/6 # scale the input to be between [-0.5, 0.5]

rho = 2
density = 0.1
subreservoir_N = 32
N = 96
rN = 50
NR = 3

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
        ESN = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density, svd_dv=0.1)
        train, test = mso.run_MSO(ESN, MSO_three, data_lengths)
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
        train, test = mso.run_MSO(ESN, MSO_three, data_lengths)
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
    DW = scan_densities()
    f = np.tanh
    DB = scan_so(N, n_subgroups, density)
    inner_size = N//n_subgroups
    TTot = TWashout + TTrain + TTest
    nrmsedf = pd.DataFrame(columns=['strain', 'stest', 'rtrain', 'rtest', 'gtrain', 'gtest', 'ctrain', 'ctest'])
    msedf = pd.DataFrame(columns=['strain', 'stest', 'rtrain', 'rtest', 'gtrain', 'gtest', 'ctrain', 'ctest'])
    rhythm_three = np.array([[1], [0, 1], [0, 0, 1]], dtype=object)
    for i in range(TRuns):
        W = rmatrix.create_restricted_esn_weights(subreservoir_N*NR, subreservoir_N, NR, DW, DB)
        standard = nymph.NymphESN(1, subreservoir_N*NR, 1, density=DW, seed=i, svd_dv=0.01)
        restricted = nymph.NymphESN(1, subreservoir_N*NR, 1, density=DW, seed=i, svd_dv=0.01)
        restricted.set_weights(W)
        gondor_encodings = [temp.TempESN_Encoding.generate_gondor_encoding()] *NR
        circuit_encodings = [temp.TempESN_Encoding.generate_circuit_encoding()] * NR
        gondor = temp.Temporal_ESN(1, subreservoir_N*NR, 1, NR, gondor_encodings, seed=i, svd_dv=0.01)
        gondor.set_rhythms(rhythm_three)
        gondor.set_weights(W)
        circuit = temp.Temporal_ESN(1, subreservoir_N*NR, 1, NR, circuit_encodings, seed=i, svd_dv=0.01)
        circuit.set_rhythms(rhythm_three)
        circuit.set_weights(W)
        snrmse, smse = mso.run_MSO_rr(standard, MSO_three, data_lengths, error="both")
        rnrmse, rmse = mso.run_MSO_rr(restricted, MSO_three, data_lengths, error="both")
        # print(f"{restricted.Wv=}")
        gnrmse, gmse = mso.run_MSO_rr(gondor, MSO_three, data_lengths, error="both")
        # print(f"{gondor.Wv=}")
        cnrmse, cmse = mso.run_MSO_rr(circuit, MSO_three, data_lengths, error="both")
        
        #nrmse test error
        nrmsedf.at[i, 'stest'] = snrmse[1]
        nrmsedf.at[i, 'rtest'] = rnrmse[1]
        nrmsedf.at[i, 'gtest'] = gnrmse[1]
        nrmsedf.at[i, 'ctest'] = cnrmse[1]
        #mse test error
        msedf.at[i, 'stest'] = smse[1]
        msedf.at[i, 'rtest'] = rmse[1]
        msedf.at[i, 'gtest'] = gmse[1]
        msedf.at[i, 'ctest'] = cmse[1]
    
    # save nrmses
    buf_nrmse = f"/home/cw1647/phd/tempESN-experiments/mso_reverse/results/nrmses/mso_three.csv"
    nrmsedf.to_csv(buf_nrmse)

    # save mses 
    buf_mse = f"/home/cw1647/phd/tempESN-experiments/mso_reverse/results/mses/mso_three.csv"
    msedf.to_csv(buf_mse)

    # compare
    
print("MSO three")
compare_distributions(N, NR)