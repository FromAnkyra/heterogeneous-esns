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

def matched_inputs(name="sleep_apnea"):
    print(name)
    nrmsedf = pd.DataFrame(columns=['heart', "chest", "blood"])
    for i in range(TRuns):
        # print(i)
        bucketesn = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True, K=3)
        input_mask = np.block([[np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 2))],
                              [np.zeros((subreservoir_size, 1)), np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 1))],
                              [np.zeros((subreservoir_size, 2)), np.ones((subreservoir_size, 1))]])
        bucketesn.set_input_weights(bucketesn.Wu * input_mask)
        bucketnrmse = sleep.run_benchmark(bucketesn, data_lengths)
        nrmsedf.at[i, 'heart'] = bucketnrmse[0]   
        nrmsedf.at[i, 'chest'] = bucketnrmse[1]
        nrmsedf.at[i, 'blood'] = bucketnrmse[2]

        buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results/{name}.csv"
    nrmsedf.to_csv(buf_nrmse)
    print("done!")
    return

matched_inputs("sleep_apnea_single_material_matched_input")
