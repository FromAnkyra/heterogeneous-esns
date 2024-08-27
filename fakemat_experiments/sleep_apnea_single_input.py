import numpy as np 
import benchmarks.sleep_apnea as sleep
import pandas as pd
import create_esns

TWashout = 200
TTest = 1000
TTrain = 500

data_lengths = (TWashout, TTest, TTrain)

TRuns = 50
fullsize = 64*3
subreservoir_size = 64

def experiment_single_timescale(name="sleep_apnea"):
    nrmsedf = pd.DataFrame(columns=['bucket', 'delayline', 'maglattice', 'mixed'])
    for mode in ["heart", "blood", "chest"]:
        for i in range(TRuns):
            # print(i)
            bucketesn = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i )
            bucketnrmse = sleep.single_input(bucketesn, data_lengths, mode)
            nrmsedf.at[i, 'bucket'] = bucketnrmse

            delaylineesn = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i)
            delaylinenrmse = sleep.single_input(delaylineesn, data_lengths, mode)
            nrmsedf.at[i, 'delayline'] = delaylinenrmse    

            maglatticeesn = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i)
            maglatticenrmse = sleep.single_input(maglatticeesn, data_lengths, mode)       
            nrmsedf.at[i, 'maglattice'] = maglatticenrmse  

            mixedesn = create_esns.create_esn_single_timescale(create_esns.mixed_matlist, fullsize, i)
            mixednrmse = sleep.single_input(mixedesn, data_lengths, mode)
            nrmsedf.at[i, 'mixed'] = mixednrmse    
        buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/results/sleep_apnea_single_input/s-no-svd-{mode}.csv"
        nrmsedf.to_csv(buf_nrmse)
        print("done!")
    return

def experiment_multi_timescale(name="sleep_apnea"):
    for mode in ["heart", "blood", "chest"]: 
        nrmsedf = pd.DataFrame(columns=['bucket', 'delayline', 'maglattice', 'mixed'])   
        for i in range(TRuns):
            # print(i)
            bucketesn = create_esns.create_esn_multi_timescale(create_esns.bucket_matlist, fullsize, i)
            bucketnrmse = sleep.single_input(bucketesn, data_lengths, mode)
            nrmsedf.at[i, 'bucket'] = bucketnrmse   

            delaylineesn = create_esns.create_esn_multi_timescale(create_esns.delayline_matlist, fullsize, i)
            delaylinenrmse = sleep.single_input(delaylineesn, data_lengths, mode)
            nrmsedf.at[i, 'delayline'] = delaylinenrmse     

            maglatticeesn = create_esns.create_esn_multi_timescale(create_esns.maglattice_matlist, fullsize, i)
            maglatticenrmse = sleep.single_input(maglatticeesn, data_lengths, mode)        
            nrmsedf.at[i, 'maglattice'] = maglatticenrmse    

            mixedesn = create_esns.create_esn_multi_timescale(create_esns.mixed_matlist, fullsize, i)
            mixednrmse = sleep.single_input(mixedesn, data_lengths, mode)
            nrmsedf.at[i, 'mixed'] = mixednrmse    
        buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/results/sleep_apnea_single_input/m-no-svd-{mode}.csv"
        nrmsedf.to_csv(buf_nrmse)
        print("done!")
    return

# normalised svds

def experiment_single_timescale_svd(name="sleep_apnea"):
    for mode in ["heart", "blood", "chest"]:    
        nrmsedf = pd.DataFrame(columns=['bucket', 'delayline', 'maglattice', 'mixed'])
        for i in range(TRuns):
            # print(i)
            bucketesn = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True)
            bucketnrmse = sleep.single_input(bucketesn, data_lengths, mode)
            nrmsedf.at[i, 'bucket'] = bucketnrmse

            delaylineesn = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True)
            delaylinenrmse = sleep.single_input(delaylineesn, data_lengths, mode)
            nrmsedf.at[i, 'delayline'] = delaylinenrmse    

            maglatticeesn = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True)
            maglatticenrmse = sleep.single_input(maglatticeesn, data_lengths, mode)     
            nrmsedf.at[i, 'maglattice'] = maglatticenrmse  

            mixedesn = create_esns.create_esn_single_timescale(create_esns.mixed_matlist, fullsize, i, normalise_svd=True)
            mixednrmse = sleep.single_input(mixedesn, data_lengths, mode)
            nrmsedf.at[i, 'mixed'] = mixednrmse    
        buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/results/sleep_apnea_single_input/s-svd-{mode}.csv"
        nrmsedf.to_csv(buf_nrmse)
        print("done!")
    return

def experiment_multi_timescale_svd(name="sleep_apnea"):
    for mode in ["heart", "blood", "chest"]:    
        nrmsedf = pd.DataFrame(columns=['bucket', 'delayline', 'maglattice', 'mixed'])
        for i in range(TRuns):
            # print(i)
            bucketesn = create_esns.create_esn_multi_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True)
            bucketnrmse = sleep.single_input(bucketesn, data_lengths, mode)
            nrmsedf.at[i, 'bucket'] = bucketnrmse   

            delaylineesn = create_esns.create_esn_multi_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True)
            delaylinenrmse = sleep.single_input(delaylineesn, data_lengths, mode)
            nrmsedf.at[i, 'delayline'] = delaylinenrmse     

            maglatticeesn = create_esns.create_esn_multi_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True)
            maglatticenrmse = sleep.single_input(maglatticeesn, data_lengths, mode)        
            nrmsedf.at[i, 'maglattice'] = maglatticenrmse    

            mixedesn = create_esns.create_esn_multi_timescale(create_esns.mixed_matlist, fullsize, i, normalise_svd=True)
            mixednrmse = sleep.single_input(mixedesn, data_lengths, mode)
            nrmsedf.at[i, 'mixed'] = mixednrmse    
        buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/results/sleep_apnea_single_input/m-svd-{mode}.csv"
        nrmsedf.to_csv(buf_nrmse)
        print("done!")
    return

# print("single timescales")
# experiment_single_timescale()
# experiment_single_timescale_svd()
print("multi timescale")
# experiment_multi_timescale()
experiment_multi_timescale_svd()
