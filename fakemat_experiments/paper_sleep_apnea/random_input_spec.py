import numpy as np 
import benchmarks.sleep_apnea as sleep
import pandas as pd
import fakemat_experiments.create_esns as create_esns

TStart = 22000
TWashout = 1000
TTrain = 3000
TTest = 1000

data_lengths = (TStart, TWashout, TTrain, TTest)

TRuns = 100
fullsize = 64*3
subreservoir_size = 64

input_map = {
    "hbrrbl": np.block([[np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 2))],
                              [np.zeros((subreservoir_size, 1)), np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 1))],
                              [np.zeros((subreservoir_size, 2)), np.ones((subreservoir_size, 1))]]),
    "hbrlbr": np.block([[np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 2))],
                              [np.zeros((subreservoir_size, 2)), np.ones((subreservoir_size, 1))],
                              [np.zeros((subreservoir_size, 1)), np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 1))]]),
    "hrrbbl": np.block([[np.zeros((subreservoir_size, 1)), np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 1))],
                              [np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 2))],
                              [np.zeros((subreservoir_size, 2)), np.ones((subreservoir_size, 1))]]),
    "hrrbbl": np.block([[np.zeros((subreservoir_size, 1)), np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 1))],
                              [np.zeros((subreservoir_size, 2)), np.ones((subreservoir_size, 1))],
                              [np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 2))]]),
    "hlrbbr": np.block([[np.zeros((subreservoir_size, 2)), np.ones((subreservoir_size, 1))],
                              [np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 2))],
                              [np.zeros((subreservoir_size, 1)), np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 1))]]),
    "hlrrbb": np.block([[np.zeros((subreservoir_size, 2)), np.ones((subreservoir_size, 1))],
                              [np.zeros((subreservoir_size, 1)), np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 1))],
                              [np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 2))]])
}

input_types = list(input_map.keys())

def experiment_single_timescale_svd(name="sleep_apnea"):
    print(name)
    nrmsedf = pd.DataFrame(columns=['heart', "chest", "blood", "input type"])
    for i in range(TRuns):
        # choose input randomly
        input_choice = np.random.choice(input_types)
        input_mask = input_map[input_choice]
        mixedesn = create_esns.create_esn_single_timescale(create_esns.mixed_matlist, fullsize, i, normalise_svd=True, K=3)
        mixedesn.set_input_weights(mixedesn.Wu * input_mask)
        mixednrmse = sleep.run_benchmark(mixedesn, data_lengths)
        nrmsedf.at[i, 'heart'] = mixednrmse[0]
        nrmsedf.at[i, 'chest'] = mixednrmse[1] 
        nrmsedf.at[i, 'blood'] = mixednrmse[2]
        nrmsedf.at[i, 'input type'] = input_choice
        buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results/{name}.csv"
    nrmsedf.to_csv(buf_nrmse)
    print("done!")
    return

# def experiment_multi_timescale_svd(name="sleep_apnea"):
#     nrmsedf = pd.DataFrame(columns=['bheart', "bchest", "bblood", 'dheart', "dchest", "dblood", 'mlheart', "mlchest", "mlblood", 'mheart', "mchest", "mblood"])
#     for i in range(TRuns):
#         # print(i)
#         bucketesn = create_esns.create_esn_multi_phase(create_esns.bucket_matlist, fullsize, i, normalise_svd=True, K=3)
#         bucketnrmse = sleep.run_benchmark(bucketesn, data_lengths)
#         nrmsedf.at[i, 'bheart'] = bucketnrmse[0]   
#         nrmsedf.at[i, 'bchest'] = bucketnrmse[1]
#         nrmsedf.at[i, 'bblood'] = bucketnrmse[2]  

#         delaylineesn = create_esns.create_esn_multi_phase(create_esns.delayline_matlist, fullsize, i, normalise_svd=True, K=3)
#         delaylinenrmse = sleep.run_benchmark(delaylineesn, data_lengths)
#         nrmsedf.at[i, 'dheart'] = delaylinenrmse[0]     
#         nrmsedf.at[i, 'dchest'] = delaylinenrmse[1]
#         nrmsedf.at[i, 'dblood'] = delaylinenrmse[2]

#         maglatticeesn = create_esns.create_esn_multi_phase(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True, K=3)
#         maglatticenrmse = sleep.run_benchmark(maglatticeesn, data_lengths)        
#         nrmsedf.at[i, 'mlheart'] = maglatticenrmse[0]   
#         nrmsedf.at[i, 'mlchest'] = maglatticenrmse[1]  
#         nrmsedf.at[i, 'mlblood'] = maglatticenrmse[2] 

#         mixedesn = create_esns.create_esn_multi_phase(create_esns.mixed_matlist, fullsize, i, normalise_svd=True, K=3)
#         mixednrmse = sleep.run_benchmark(mixedesn, data_lengths)
#         nrmsedf.at[i, 'mheart'] = mixednrmse[0]
#         nrmsedf.at[i, 'mchest'] = mixednrmse[1] 
#         nrmsedf.at[i, 'mblood'] = mixednrmse[2]
#         buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/results/normalised_total_svd/multi_timescale/{name}.csv"
#     nrmsedf.to_csv(buf_nrmse)
#     print("done!")
#     return

experiment_single_timescale_svd("sleep_apnea_random_input_matching.csv")