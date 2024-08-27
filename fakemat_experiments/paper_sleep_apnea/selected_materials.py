import numpy as np 
import benchmarks.sleep_apnea as sleep
import pandas as pd
import fakemat_experiments.create_esns as create_esns
import fakemat_experiments.bucket as bucket
import fakemat_experiments.delayline as delayline
import fakemat_experiments.maglattice as maglattice

TStart = 22000
TWashout = 1000
TTrain = 3000
TTest = 1000

data_lengths = (TStart, TWashout, TTrain, TTest)

TRuns = 100
fullsize = 64*3
subreservoir_size = 64

matched_materials = {
    0: bucket.Bucket(subreservoir_size, 0, 0),
    1: maglattice.MagLattice(subreservoir_size, 1, 0),
    2: maglattice.MagLattice(subreservoir_size, 2, 0)
}

def matched_inputs(name="sleep_apnea"):
    print(name)
    nrmsedf = pd.DataFrame(columns=['heart', "chest", "blood"])
    for i in range(TRuns):
        # print(i)
        matchedesn = create_esns.create_esn_single_timescale(matched_materials, fullsize, i, normalise_svd=True, K=3)
        input_mask = np.block([[np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 2))],
                              [np.zeros((subreservoir_size, 1)), np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 1))],
                              [np.zeros((subreservoir_size, 2)), np.ones((subreservoir_size, 1))]])
        matchedesn.set_input_weights(matchedesn.Wu * input_mask)
        matchednrmse = sleep.run_benchmark(matchedesn, data_lengths)
        nrmsedf.at[i, 'heart'] = matchednrmse[0]   
        nrmsedf.at[i, 'chest'] = matchednrmse[1]
        nrmsedf.at[i, 'blood'] = matchednrmse[2]

        buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results/{name}.csv"
    nrmsedf.to_csv(buf_nrmse)
    print("done!")
    return

def unspecified_inputs(name):
    print(name)
    nrmsedf = pd.DataFrame(columns=['heart', "chest", "blood"])
    for i in range(TRuns):
        # print(i)
        matchedesn = create_esns.create_esn_single_timescale(matched_materials, fullsize, i, normalise_svd=True, K=3)
        matchednrmse = sleep.run_benchmark(matchedesn, data_lengths)
        nrmsedf.at[i, 'heart'] = matchednrmse[0]   
        nrmsedf.at[i, 'chest'] = matchednrmse[1]
        nrmsedf.at[i, 'blood'] = matchednrmse[2]

        buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results/{name}.csv"
    nrmsedf.to_csv(buf_nrmse)
    print("done!")
    return


matched_inputs("sleep_apnea_selected_material_matched_input")
unspecified_inputs("sleep_apnea_selected_material_unspecified_input")


