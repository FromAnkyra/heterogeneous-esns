import numpy as np
import benchmarks.mso as mso
import pandas as pd
import sys
import fakemat_experiments.create_esns as create_esns

resolution_4 = np.linspace(0, 500, 4000)

MSO_list = [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value]
MSO_three = mso.generate_MSO(resolution_4, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value])
MSO_three = MSO_three/6

TWashout = 100
TTest = 200
TTrain = 800

data_lengths = (TWashout, TTrain, TTest)

TRuns = 50
fullsize = 64*3
subreservoir_size = 64

input_mask = np.block([[np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 2))],
                              [np.zeros((subreservoir_size, 1)), np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 1))],
                              [np.zeros((subreservoir_size, 2)), np.ones((subreservoir_size, 1))]])

def run_experiment():
    single_input_df = pd.DataFrame(columns=['bucket', 'ring', 'lattice'])
    multi_input_df = pd.DataFrame(columns=['bucket', 'ring', 'lattice'])
    print("start")
    for i in range(TRuns):
        print(i)
        bucketsingleesn = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True)
        bucketsinglenrmse = mso.run_MSO_rr(bucketsingleesn, MSO_three, data_lengths, error="nrmse")        
        single_input_df.at[i, 'bucket'] = bucketsinglenrmse[1]  

        bucketmultiesn = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True, K=3)
        bucketmultinrmse = mso.run_MSO_multi_input(bucketmultiesn, MSO_list, resolution_4, data_lengths, input_mapping=True, input_mask=input_mask)         
        multi_input_df.at[i, 'bucket'] = bucketmultinrmse[1] 

        ringsingleesn = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True)
        ringsinglenrmse = mso.run_MSO_rr(ringsingleesn, MSO_three, data_lengths, error="nrmse")        
        single_input_df.at[i, 'ring'] = ringsinglenrmse[1]   

        ringmultiesn = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True, K=3)
        ringmultinrmse = mso.run_MSO_multi_input(ringmultiesn, MSO_list, resolution_4, data_lengths, input_mapping=True, input_mask=input_mask)         
        multi_input_df.at[i, 'ring'] = ringmultinrmse[1]   

        latticesingleesn = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True)
        latticesinglenrmse = mso.run_MSO_rr(latticesingleesn, MSO_three, data_lengths, error="nrmse")        
        single_input_df.at[i, 'lattice'] = latticesinglenrmse[1] 

        latticemultiesn = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True, K=3)
        latticemultinrmse = mso.run_MSO_multi_input(latticemultiesn, MSO_list, resolution_4, data_lengths, input_mapping=True, input_mask=input_mask)         
        multi_input_df.at[i, 'lattice'] = latticemultinrmse[1]    
    
    buf_single = "/home/cw1647/phd/het_reservoir_experiments/mso/prelim_experiments/multi-v-single-input/results/single-input.csv"    
    single_input_df.to_csv(buf_single)

    buf_multi = "/home/cw1647/phd/het_reservoir_experiments/mso/prelim_experiments/multi-v-single-input/results/multi-input.csv"    
    multi_input_df.to_csv(buf_multi)
    print("done!")
    return

def experiment_all_to_all():
    single_input_df = pd.DataFrame(columns=['bucket', 'ring', 'lattice'])
    multi_input_df = pd.DataFrame(columns=['bucket', 'ring', 'lattice'])
    print("start")
    for i in range(TRuns):
        print(i)
        bucketsingleesn = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True)
        bucketsinglenrmse = mso.run_MSO_rr(bucketsingleesn, MSO_three, data_lengths, error="nrmse")        
        single_input_df.at[i, 'bucket'] = bucketsinglenrmse[1]  

        bucketmultiesn = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True, K=3)
        bucketmultinrmse = mso.run_MSO_multi_input(bucketmultiesn, MSO_list, resolution_4, data_lengths)         
        multi_input_df.at[i, 'bucket'] = bucketmultinrmse[1] 

        ringsingleesn = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True)
        ringsinglenrmse = mso.run_MSO_rr(ringsingleesn, MSO_three, data_lengths, error="nrmse")        
        single_input_df.at[i, 'ring'] = ringsinglenrmse[1]   

        ringmultiesn = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True, K=3)
        ringmultinrmse = mso.run_MSO_multi_input(ringmultiesn, MSO_list, resolution_4, data_lengths)         
        multi_input_df.at[i, 'ring'] = ringmultinrmse[1]   

        latticesingleesn = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True)
        latticesinglenrmse = mso.run_MSO_rr(latticesingleesn, MSO_three, data_lengths, error="nrmse")        
        single_input_df.at[i, 'lattice'] = latticesinglenrmse[1] 

        latticemultiesn = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True, K=3)
        latticemultinrmse = mso.run_MSO_multi_input(latticemultiesn, MSO_list, resolution_4, data_lengths)         
        multi_input_df.at[i, 'lattice'] = latticemultinrmse[1]    
    
    # buf_single = "/home/cw1647/phd/het_reservoir_experiments/mso/prelim_experiments/multi-v-single-input/results/single-input.csv"    
    # single_input_df.to_csv(buf_single)

    buf_multi = "/home/cw1647/phd/het_reservoir_experiments/mso/prelim_experiments/multi-v-single-input/results/multi-input-all-to-all.csv"    
    multi_input_df.to_csv(buf_multi)
    print("done!")
    return


run_experiment()
# experiment_all_to_all()
