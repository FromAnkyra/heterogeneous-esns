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

def experiment_single_timescale_svd(name="sleep_apnea"):
    print(name)
    nrmsedf = pd.DataFrame(columns=['bheart', "bchest", "bblood", 'dheart', "dchest", "dblood", 'mlheart', "mlchest", "mlblood", 'mheart', "mchest", "mblood"])
    for i in range(TRuns):
        print(i)
        bheartesn = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True)
        bheartnrmse = sleep.single_input(bheartesn, data_lengths, "heart")

        bchestesn = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True)
        bchestnrmse = sleep.single_input(bchestesn, data_lengths, "chest")

        bbloodesn = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True)
        bbloodnrmse = sleep.single_input(bbloodesn, data_lengths, "blood")

        nrmsedf.at[i, 'bheart'] = bheartnrmse   
        nrmsedf.at[i, 'bchest'] = bchestnrmse
        nrmsedf.at[i, 'bblood'] = bbloodnrmse

        dheartesn = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True)
        dheartnrmse = sleep.single_input(dheartesn, data_lengths, "heart")

        dchestesn = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True)
        dchestnrmse = sleep.single_input(dchestesn, data_lengths, "chest")

        dbloodesn = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True)
        dbloodnrmse = sleep.single_input(dbloodesn, data_lengths, "blood")

        nrmsedf.at[i, 'dheart'] = dheartnrmse     
        nrmsedf.at[i, 'dchest'] = dchestnrmse
        nrmsedf.at[i, 'dblood'] = dbloodnrmse

        mlheartesn = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True)
        mlheartnrmse = sleep.single_input(mlheartesn, data_lengths, "heart")

        mlchestesn = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True)
        mlchestnrmse = sleep.single_input(mlchestesn, data_lengths, "chest")

        mlbloodesn = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True)
        mlbloodnrmse = sleep.single_input(mlbloodesn, data_lengths, "blood")

        nrmsedf.at[i, 'mlheart'] = mlheartnrmse   
        nrmsedf.at[i, 'mlchest'] = mlchestnrmse  
        nrmsedf.at[i, 'mlblood'] = mlbloodnrmse 

        mheartesn = create_esns.create_esn_single_timescale(create_esns.mixed_matlist, fullsize, i, normalise_svd=True)
        mheartnrmse = sleep.single_input(mheartesn, data_lengths, "heart")

        mchestesn = create_esns.create_esn_single_timescale(create_esns.mixed_matlist, fullsize, i, normalise_svd=True)
        mchestnrmse = sleep.single_input(mchestesn, data_lengths, "chest")

        mbloodesn = create_esns.create_esn_single_timescale(create_esns.mixed_matlist, fullsize, i, normalise_svd=True)
        mbloodnrmse = sleep.single_input(mbloodesn, data_lengths, "blood")
        nrmsedf.at[i, 'mheart'] = mheartnrmse
        nrmsedf.at[i, 'mchest'] = mchestnrmse
        nrmsedf.at[i, 'mblood'] = mbloodnrmse
        buf_nrmse = f"/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results/{name}.csv"
    nrmsedf.to_csv(buf_nrmse)
    print("done!")
    return

experiment_single_timescale_svd("sleep_apnea_separated_input")