# train reservoirs on time series predictions
# laser dataset from ???
# sunspot dataset from https://machinelearningmastery.com/time-series-datasets-for-machine-learning/

import numpy as np
import pandas as pd

import reservoir


def get_u():
    # read in dataset from csv
    # sunspot data has 2820 entries (rows 0--2819)
    df = pd.read_csv('monthly-sunspots.csv')
    data = df['Sunspots'].to_numpy()
    # normalise data to range [0,0.5]
    max = np.amax(data)
    data = data / (2 * max)
    return data.tolist()


# ===================================================================
# ===================================================================
# reservoir parameters
rho = 2
density = 0.2
N = 50  # no of nodes

Twashout = 320
Ttrain = 500
Ttest = 2000
Ttot = Twashout + Ttrain + Ttest

Nruns = 50  # number of train/test runs per exptl configuration
config = ['flat', 'physical']
config_name = ['(a)', '(b)']

col_names = [['train ' + str(conf), 'test ' + str(conf)] for conf in config_name]
col_names = [item for sublist in col_names for item in sublist]
errordf = pd.DataFrame(columns=col_names)

input = get_u()

for run in range(Nruns):
    print(run)
    np.random.seed(run)

    system = pd.DataFrame(columns=['u', 'x', 'v', 'vtarget'])  # holds all the time states

    system['u'] = input
    # shift = how far in the future to predict, indep of config
    # add copies of last value to end to pad
    shift = 1
    system['vtarget'] = input[shift:] + [input[-1]] * shift

    # reservoir random weights
    rc = reservoir.Reservoir(N, rho=rho, density=density, seed=run)

    for c, conf in enumerate(config):
        rc.set_config(conf)

        # run the reservoir using  input
        rc.run_reservoir(Ttot, system)
        # train reservoir
        Wv = rc.train_reservoir(Twashout, Ttrain, system)

        # get training and testing errors
        rc.get_reservoir_output(Twashout + Ttrain + Ttest, Wv, system)

        train_str = 'train ' + str(config_name[c])
        test_str = 'test ' + str(config_name[c])
        errordf.at[run, train_str] = rc.get_error(Twashout, Ttrain, system)
        errordf.at[run, test_str] = rc.get_error(Twashout + Ttrain, Ttest, system)

reservoir.plot_results(system, errordf, Twashout, Ttrain, Ttest, 'sunspot.pdf', boxplot_only=True)
