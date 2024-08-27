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

TRuns = 1

NARMA = 10

def compare_distributions(N):
    # density = scan_densities(N)
    density = 0.1
    f = np.tanh
    # p = scan_p(N, n_subgroups, density)
    # within, outwith = find_distribution(n_subgroups, p, density)
    # inner_size = N//n_subgroups
    TTot = TWashout + TTrain + TTest
    errordf = pd.DataFrame(columns=['strain', 'stest', 'rtrain', 'rtest'])
    # compare
    for i in range(50):
        standard = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        restricted = nymph.NymphESN(1, N, 1, seed=i, f=f, rho=2, density=density)
        # W = rmatrix.create_restricted_esn_weights(N, inner_size, n_subgroups, within, outwith)
        input = get_u(T=TTot, seed=i)
        vtarget = run_narma(NARMA, TTot, input, debug=False)
        # vtarget = np.hstack((np.array([0, 0, 0]), input[:-3]))
        vtarget_np = np.array(vtarget)
        standard.set_data_lengths(TWashout, TTrain, TTest)
        restricted.set_data_lengths(TWashout, TTrain, TTest)
        standard.set_input_stream(input)
        restricted.set_input_stream(input)
        standard.run_full()
        restricted.run_full()
        restricted.train_reservoir(vtarget_np[TWashout:TWashout+TTrain])
        standard.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])
        # restricted.train_ridge_regression(vtarget_np[TWashout:TWashout+TTrain])
        standard.get_output()
        restricted.get_output()
        strain, stest = standard.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        rtrain, rtest = restricted.get_error(vtarget_np, errorfunc.ErrorFuncs.nrmse)
        errordf.at[i, 'strain'] = strain
        errordf.at[i, 'stest'] = stest
        errordf.at[i, 'rtrain'] = rtrain
        errordf.at[i, 'rtest'] = rtest
    buf = f"/home/cw1647/phd/ridge_regression/rr-test.csv"
    bufvis = f"/home/cw1647/phd/ridge_regression/rr-test.png"
    errordf.to_csv(buf)
    vis.ErrorVis.vis(errordf, bufvis)
    #save that to a file somewhere
    return

for size in [100]:
    print(size)
    compare_distributions(size)




