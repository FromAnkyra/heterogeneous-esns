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
    nrmsedf = pd.DataFrame(columns=['bheart', "bchest", "bblood", 'dheart', "dchest", "dblood", 'mlheart', "mlchest", "mlblood", 'mheart', "mchest", "mblood"])
    for i in range(TRuns):
        # print(i)
        bucketesn = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, K=3)
        bucketnrmse = sleep.run_benchmark(bucketesn, data_lengths)
        nrmsedf.at[i, 'bheart'] = bucketnrmse[0]   
        nrmsedf.at[i, 'bchest'] = bucketnrmse[1]
        nrmsedf.at[i, 'bblood'] = bucketnrmse[2]

        delaylineesn = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i, K=3)
        delaylinenrmse = sleep.run_benchmark(delaylineesn, data_lengths)
        nrmsedf.at[i, 'dheart'] = delaylinenrmse[0]     
        nrmsedf.at[i, 'dchest'] = delaylinenrmse[1]
        nrmsedf.at[i, 'dblood'] = delaylinenrmse[2]    

        maglatticeesn = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i, K=3)
        maglatticenrmse = sleep.run_benchmark(maglatticeesn, data_lengths)        
        nrmsedf.at[i, 'mlheart'] = maglatticenrmse[0]   
        nrmsedf.at[i, 'mlchest'] = maglatticenrmse[1]  
        nrmsedf.at[i, 'mlblood'] = maglatticenrmse[2] 

        mixedesn = create_esns.create_esn_single_timescale(create_esns.mixed_matlist, fullsize, i, K=3)
        mixednrmse = sleep.run_benchmark(mixedesn, data_lengths)
        nrmsedf.at[i, 'mheart'] = mixednrmse[0]
        nrmsedf.at[i, 'mchest'] = mixednrmse[1] 
        nrmsedf.at[i, 'mblood'] = mixednrmse[2] 
    buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/results/single_timescale/{name}.csv"
    nrmsedf.to_csv(buf_nrmse)
    print("done!")
    return

def experiment_multi_timescale(name="sleep_apnea"):
    nrmsedf = pd.DataFrame(columns=['bheart', "bchest", "bblood", 'dheart', "dchest", "dblood", 'mlheart', "mlchest", "mlblood", 'mheart', "mchest", "mblood"])
    for i in range(TRuns):
        # print(i)
        bucketesn = create_esns.create_esn_multi_phase(create_esns.bucket_matlist, fullsize, i, K=3)
        bucketnrmse = sleep.run_benchmark(bucketesn, data_lengths)
        nrmsedf.at[i, 'bheart'] = bucketnrmse[0]   
        nrmsedf.at[i, 'bchest'] = bucketnrmse[1]
        nrmsedf.at[i, 'bblood'] = bucketnrmse[2]   
   

        delaylineesn = create_esns.create_esn_multi_phase(create_esns.delayline_matlist, fullsize, i, K=3)
        delaylinenrmse = sleep.run_benchmark(delaylineesn, data_lengths)
        nrmsedf.at[i, 'dheart'] = delaylinenrmse[0]     
        nrmsedf.at[i, 'dchest'] = delaylinenrmse[1]
        nrmsedf.at[i, 'dblood'] = delaylinenrmse[2]
        
        maglatticeesn = create_esns.create_esn_multi_phase(create_esns.maglattice_matlist, fullsize, i, K=3)
        maglatticenrmse = sleep.run_benchmark(maglatticeesn, data_lengths)        
        nrmsedf.at[i, 'mlheart'] = maglatticenrmse[0]   
        nrmsedf.at[i, 'mlchest'] = maglatticenrmse[1]  
        nrmsedf.at[i, 'mlblood'] = maglatticenrmse[2]   
 

        mixedesn = create_esns.create_esn_multi_phase(create_esns.mixed_matlist, fullsize, i, K=3)
        mixednrmse = sleep.run_benchmark(mixedesn, data_lengths)
        nrmsedf.at[i, 'mheart'] = mixednrmse[0]
        nrmsedf.at[i, 'mchest'] = mixednrmse[1] 
        nrmsedf.at[i, 'mblood'] = mixednrmse[2]     
    buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/results/multi_timescale/{name}.csv"
    nrmsedf.to_csv(buf_nrmse)
    print("done!")
    return

# normalised svds

def experiment_single_timescale_svd(name="sleep_apnea"):
    nrmsedf = pd.DataFrame(columns=['bheart', "bchest", "bblood", 'dheart', "dchest", "dblood", 'mlheart', "mlchest", "mlblood", 'mheart', "mchest", "mblood"])
    for i in range(TRuns):
        # print(i)
        bucketesn = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True, K=3)
        bucketnrmse = sleep.run_benchmark(bucketesn, data_lengths)
        nrmsedf.at[i, 'bheart'] = bucketnrmse[0]   
        nrmsedf.at[i, 'bchest'] = bucketnrmse[1]
        nrmsedf.at[i, 'bblood'] = bucketnrmse[2]

        delaylineesn = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True, K=3)
        delaylinenrmse = sleep.run_benchmark(delaylineesn, data_lengths)
        nrmsedf.at[i, 'dheart'] = delaylinenrmse[0]     
        nrmsedf.at[i, 'dchest'] = delaylinenrmse[1]
        nrmsedf.at[i, 'dblood'] = delaylinenrmse[2]

        maglatticeesn = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True, K=3)
        maglatticenrmse = sleep.run_benchmark(maglatticeesn, data_lengths)        
        nrmsedf.at[i, 'mlheart'] = maglatticenrmse[0]   
        nrmsedf.at[i, 'mlchest'] = maglatticenrmse[1]  
        nrmsedf.at[i, 'mlblood'] = maglatticenrmse[2] 

        mixedesn = create_esns.create_esn_single_timescale(create_esns.mixed_matlist, fullsize, i, normalise_svd=True, K=3)
        mixednrmse = sleep.run_benchmark(mixedesn, data_lengths)
        nrmsedf.at[i, 'mheart'] = mixednrmse[0]
        nrmsedf.at[i, 'mchest'] = mixednrmse[1] 
        nrmsedf.at[i, 'mblood'] = mixednrmse[2]
        buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/results/normalised_total_svd/single_timescale/{name}.csv"
    nrmsedf.to_csv(buf_nrmse)
    print("done!")
    return

def experiment_multi_timescale_svd(name="sleep_apnea"):
    nrmsedf = pd.DataFrame(columns=['bheart', "bchest", "bblood", 'dheart', "dchest", "dblood", 'mlheart', "mlchest", "mlblood", 'mheart', "mchest", "mblood"])
    for i in range(TRuns):
        # print(i)
        bucketesn = create_esns.create_esn_multi_phase(create_esns.bucket_matlist, fullsize, i, normalise_svd=True, K=3)
        bucketnrmse = sleep.run_benchmark(bucketesn, data_lengths)
        nrmsedf.at[i, 'bheart'] = bucketnrmse[0]   
        nrmsedf.at[i, 'bchest'] = bucketnrmse[1]
        nrmsedf.at[i, 'bblood'] = bucketnrmse[2]  

        delaylineesn = create_esns.create_esn_multi_phase(create_esns.delayline_matlist, fullsize, i, normalise_svd=True, K=3)
        delaylinenrmse = sleep.run_benchmark(delaylineesn, data_lengths)
        nrmsedf.at[i, 'dheart'] = delaylinenrmse[0]     
        nrmsedf.at[i, 'dchest'] = delaylinenrmse[1]
        nrmsedf.at[i, 'dblood'] = delaylinenrmse[2]

        maglatticeesn = create_esns.create_esn_multi_phase(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True, K=3)
        maglatticenrmse = sleep.run_benchmark(maglatticeesn, data_lengths)        
        nrmsedf.at[i, 'mlheart'] = maglatticenrmse[0]   
        nrmsedf.at[i, 'mlchest'] = maglatticenrmse[1]  
        nrmsedf.at[i, 'mlblood'] = maglatticenrmse[2] 

        mixedesn = create_esns.create_esn_multi_phase(create_esns.mixed_matlist, fullsize, i, normalise_svd=True, K=3)
        mixednrmse = sleep.run_benchmark(mixedesn, data_lengths)
        nrmsedf.at[i, 'mheart'] = mixednrmse[0]
        nrmsedf.at[i, 'mchest'] = mixednrmse[1] 
        nrmsedf.at[i, 'mblood'] = mixednrmse[2]
        buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/results/normalised_total_svd/multi_timescale/{name}.csv"
    nrmsedf.to_csv(buf_nrmse)
    print("done!")
    return

print("single timescales")
experiment_single_timescale()
experiment_single_timescale_svd()
print("multi timescale")
experiment_multi_timescale()
experiment_multi_timescale_svd()