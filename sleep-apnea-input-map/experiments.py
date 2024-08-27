import numpy as np 
import benchmarks.sleep_apnea as sleep
import pandas as pd
import fakemat_experiments.create_esns as create_esns
import NymphESN.nymphesn as nymph
import NymphESN.restrictedmatrix as rmatrix
import benchmarks.sleep_apnea as sleep


TStart = 22000
TWashout = 1000
TTrain = 3000
TTest = 1000

data_lengths = (TStart, TWashout, TTrain, TTest)

TRuns = 100
fullsize = 64*3
subreservoir_size = 64

def experiment_classical(i, nrmsedf):
    ESN = nymph.NymphESN(3, fullsize, 1, seed=i, f=np.tanh, rho=2, density=0.4)
    nrmse = sleep.run_benchmark(ESN, data_lengths)
    nrmsedf.at[i, 'heart'] = nrmse[0]   
    nrmsedf.at[i, 'chest'] = nrmse[1]
    nrmsedf.at[i, 'blood'] = nrmse[2]
    return

def experiment_all_to_all(i, nrmsedf):
    ESN = nymph.NymphESN(3, fullsize, 1, seed=i, f=np.tanh, rho=2, density=0.4) 
    W = rmatrix.create_restricted_esn_weights(fullsize, subreservoir_size, 3, 0.4, 0.1) 
    ESN.set_weights(W)
    nrmse = sleep.run_benchmark(ESN, data_lengths)
    nrmsedf.at[i, 'heart'] = nrmse[0]   
    nrmsedf.at[i, 'chest'] = nrmse[1]
    nrmsedf.at[i, 'blood'] = nrmse[2]
    return

def experiment_one_to_one(i, nrmsedf):
    ESN = nymph.NymphESN(3, fullsize, 1, seed=i, f=np.tanh, rho=2, density=0.4) 
    W = rmatrix.create_restricted_esn_weights(fullsize, subreservoir_size, 3, 0.4, 0.1) 
    input_mask = np.block([[np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 2))],
                              [np.zeros((subreservoir_size, 1)), np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 1))],
                              [np.zeros((subreservoir_size, 2)), np.ones((subreservoir_size, 1))]])
    ESN.set_weights(W)
    ESN.set_input_weights(ESN.Wu*input_mask)
    nrmse = sleep.run_benchmark(ESN, data_lengths)
    nrmsedf.at[i, 'heart'] = nrmse[0]   
    nrmsedf.at[i, 'chest'] = nrmse[1]
    nrmsedf.at[i, 'blood'] = nrmse[2]
    return

def run_all():
    print("start")
    classical = pd.DataFrame(columns=["heart", "chest", "blood"])
    all_to_all = pd.DataFrame(columns=["heart", "chest", "blood"])
    one_to_one = pd.DataFrame(columns=["heart", "chest", "blood"])
    for i in range(TRuns):
        print(i)
        experiment_classical(i, classical)
        experiment_all_to_all(i, all_to_all)
        experiment_one_to_one(i, one_to_one)
    classical.to_csv("/home/cw1647/phd/sleep-apnea-input-map/results/classical")
    all_to_all.to_csv("/home/cw1647/phd/sleep-apnea-input-map/results/all_to_all")
    one_to_one.to_csv("/home/cw1647/phd/sleep-apnea-input-map/results/one_to_one")
    return

run_all()