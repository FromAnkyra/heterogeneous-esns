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
import create_esns

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

TRuns = 50
fullsize = 64*3
subreservoir_size = 64

NARMA = 10

def experiment_single_timescale(name="NARMA"):

    f = np.tanh
    TTot = TWashout + TTrain + TTest
    nrmsedf = pd.DataFrame(columns=['bucket', 'delayline', 'maglattice', 'mixed'])
    # compare
    for i in range(TRuns):
        bucketesn = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i)
        delaylineesn = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i)
        maglatticeesn = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i)
        mixedesn = create_esns.create_esn_single_timescale(create_esns.mixed_matlist, fullsize, i)
        input = get_u(T=TTot, seed=i)
        vtarget = run_narma(NARMA, TTot, input, debug=False)
        # vtarget = np.hstack((np.array([0, 0, 0]), input[:-3]))
        vtarget_np = np.array(vtarget)
        bucketesn.set_data_lengths(TWashout, TTrain, TTest)
        delaylineesn.set_data_lengths(TWashout, TTrain, TTest)
        maglatticeesn.set_data_lengths(TWashout, TTrain, TTest)
        mixedesn.set_data_lengths(TWashout, TTrain, TTest)

        bucketesn.set_input_stream(input)
        delaylineesn.set_input_stream(input)
        maglatticeesn.set_input_stream(input)
        mixedesn.set_input_stream(input)

        bucketesn.run_full()
        delaylineesn.run_full()
        maglatticeesn.run_full()
        mixedesn.run_full()

        bucketesn.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])
        delaylineesn.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])
        maglatticeesn.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])
        mixedesn.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])

        bucketesn.get_output()
        delaylineesn.get_output()
        maglatticeesn.get_output()
        mixedesn.get_output()

        
        _, bucketnrmse = bucketesn.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        _, delaylinenrmse = delaylineesn.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        _, maglatticenrmse = maglatticeesn.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        _, mixednrmse = mixedesn.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)

        nrmsedf.at[i, 'bucket'] = bucketnrmse   
        nrmsedf.at[i, 'delayline'] = delaylinenrmse   
        nrmsedf.at[i, 'maglattice'] = maglatticenrmse   
        nrmsedf.at[i, 'mixed'] = mixednrmse   

    buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/results/single_timescale/{name}.csv"    # bufvis = f"/home/cw1647/phd/UCNC-experiments/results/narma-physical-again/size-{N}-subgroups-{n_subgroups}-sw-{density}-so-{so}.svg"
    nrmsedf.to_csv(buf_nrmse)
    print("done!")
    return

def experiment_multi_timescale(name="NARMA"):
    f = np.tanh
    TTot = TWashout + TTrain + TTest
    nrmsedf = pd.DataFrame(columns=['bucket', 'delayline', 'maglattice', 'mixed'])
    # compare
    for i in range(TRuns):
        bucketesn = create_esns.create_esn_multi_phase(create_esns.bucket_matlist, fullsize, i)
        delaylineesn = create_esns.create_esn_multi_phase(create_esns.delayline_matlist, fullsize, i)
        maglatticeesn = create_esns.create_esn_multi_phase(create_esns.maglattice_matlist, fullsize, i)
        mixedesn = create_esns.create_esn_multi_phase(create_esns.mixed_matlist, fullsize, i)
        input = get_u(T=TTot, seed=i)
        vtarget = run_narma(NARMA, TTot, input, debug=False)
        # vtarget = np.hstack((np.array([0, 0, 0]), input[:-3]))
        vtarget_np = np.array(vtarget)
        bucketesn.set_data_lengths(TWashout, TTrain, TTest)
        delaylineesn.set_data_lengths(TWashout, TTrain, TTest)
        maglatticeesn.set_data_lengths(TWashout, TTrain, TTest)
        mixedesn.set_data_lengths(TWashout, TTrain, TTest)

        bucketesn.set_input_stream(input)
        delaylineesn.set_input_stream(input)
        maglatticeesn.set_input_stream(input)
        mixedesn.set_input_stream(input)

        bucketesn.run_full()
        delaylineesn.run_full()
        maglatticeesn.run_full()
        mixedesn.run_full()

        bucketesn.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])
        delaylineesn.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])
        maglatticeesn.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])
        mixedesn.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])

        bucketesn.get_output()
        delaylineesn.get_output()
        maglatticeesn.get_output()
        mixedesn.get_output()

        
        _, bucketnrmse = bucketesn.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        _, delaylinenrmse = delaylineesn.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        _, maglatticenrmse = maglatticeesn.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        _, mixednrmse = mixedesn.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)

        nrmsedf.at[i, 'bucket'] = bucketnrmse
        nrmsedf.at[i, 'delayline'] = delaylinenrmse   
        nrmsedf.at[i, 'maglattice'] = maglatticenrmse   
        nrmsedf.at[i, 'mixed'] = mixednrmse   

    buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/results/multi_timescale/{name}.csv"    # bufvis = f"/home/cw1647/phd/UCNC-experiments/results/narma-physical-again/size-{N}-subgroups-{n_subgroups}-sw-{density}-so-{so}.svg"
    nrmsedf.to_csv(buf_nrmse)
    print("done!")
    #save that to a file somewhere
    return

# print("single_timescale")
# experiment_single_timescale()
print("multi_timescale")
experiment_multi_timescale()



