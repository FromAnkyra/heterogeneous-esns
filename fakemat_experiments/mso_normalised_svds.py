import numpy as np
import benchmarks.mso as mso
import pandas as pd
import sys
import create_esns

resolution_4 = np.linspace(0, 500, 4000)
MSO_eight = mso.generate_MSO(resolution_4, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value, mso.MSO.four.value, mso.MSO.five.value, mso.MSO.six.value, mso.MSO.seven.value, mso.MSO.eight.value])
MSO_eight = MSO_eight/16 # scale the input to be between [-0.5, 0.5]
MSO_two = mso.generate_MSO(resolution_4, [mso.MSO.one.value, mso.MSO.two.value])
MSO_two = MSO_two/4
MSO_four = mso.generate_MSO(resolution_4, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value, mso.MSO.four.value])
MSO_four = MSO_four/8
TWashout = 100
TTest = 200
TTrain = 400

data_lengths = (TWashout, TTest, TTrain)

TRuns = 50
fullsize = 64*3
subreservoir_size = 64

def experiment_single_timescale(MSO, name):
    nrmsedf = pd.DataFrame(columns=['bucket', 'delayline', 'maglattice', 'mixed'])
    for i in range(TRuns):
        # print(i)
        bucketesn = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True)
        bucketnrmse, _ = mso.run_MSO_rr(bucketesn, MSO, data_lengths, error="both")        
        nrmsedf.at[i, 'bucket'] = bucketnrmse[1]   

        delaylineesn = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True)
        delaylinenrmse, _ = mso.run_MSO_rr(delaylineesn, MSO, data_lengths, error="both")        
        nrmsedf.at[i, 'delayline'] = delaylinenrmse[1]     

        maglatticeesn = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True)
        maglatticenrmse, _ = mso.run_MSO_rr(maglatticeesn, MSO, data_lengths, error="both")        
        nrmsedf.at[i, 'maglattice'] = maglatticenrmse[1]    

        mixedesn = create_esns.create_esn_single_timescale(create_esns.mixed_matlist, fullsize, i, normalise_svd=True)
        mixednrmse, _ = mso.run_MSO_rr(mixedesn, MSO, data_lengths, error="both")
        nrmsedf.at[i, 'mixed'] = mixednrmse[1]    
    buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/results/normalised_total_svd/single_timescale/{name}.csv"
    nrmsedf.to_csv(buf_nrmse)
    print("done!")
    return

def experiment_multi_timescale(MSO, name):
    nrmsedf = pd.DataFrame(columns=['bucket', 'delayline', 'maglattice', 'mixed'])
    for i in range(TRuns):
        bucketesn = create_esns.create_esn_multi_phase(create_esns.bucket_matlist, fullsize, i, normalise_svd=True)
        bucketnrmse, _ = mso.run_MSO_rr(bucketesn, MSO, data_lengths, error="both")        
        nrmsedf.at[i, 'bucket'] = bucketnrmse[1]   

        delaylineesn = create_esns.create_esn_multi_phase(create_esns.delayline_matlist, fullsize, i, normalise_svd=True)
        delaylinenrmse, _ = mso.run_MSO_rr(delaylineesn, MSO, data_lengths, error="both")        
        nrmsedf.at[i, 'delayline'] = delaylinenrmse[1]     

        maglatticeesn = create_esns.create_esn_multi_phase(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True)
        maglatticenrmse, _ = mso.run_MSO_rr(maglatticeesn, MSO, data_lengths, error="both")        
        nrmsedf.at[i, 'maglattice'] = maglatticenrmse[1]        
    
        mixedesn = create_esns.create_esn_multi_phase(create_esns.mixed_matlist, fullsize, i, normalise_svd=True)
        mixednrmse, _ = mso.run_MSO_rr(mixedesn, MSO, data_lengths, error="both")
        nrmsedf.at[i, 'mixed'] = mixednrmse[1] 
    buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/results/normalised_total_svd/multi_timescale/{name}.csv"
    nrmsedf.to_csv(buf_nrmse)
    print("done!")
    return
# print("single timescales")
# experiment_single_timescale(MSO_two, "mso_two")
# experiment_single_timescale(MSO_four, "mso_four")
# experiment_single_timescale(MSO_eight, "mso_eight")
print("multi timescales")
experiment_multi_timescale(MSO_two, "mso_two")
experiment_multi_timescale(MSO_four, "mso_four")
experiment_multi_timescale(MSO_eight, "mso_eight")

