import numpy as np
import benchmarks.sleep_apnea as sleep
import NymphESN.nymphesn as nymph
import NymphESN.restrictedmatrix as rmatrix
import pandas as pd
import sys
import fakemat_experiments.create_esns as create_esns
import fakemat_experiments.bucket as bucket
import fakemat_experiments.delayline as ring
import fakemat_experiments.maglattice as lattice

TStart = 22000
TWashout = 1000
TTrain = 3000
TTest = 1000

data_lengths = (TStart, TWashout, TTrain, TTest)

TRuns = 50 
fullsize = 64*3
subreservoir_size = 64

fast = [1, 1, 1, 1, 1, 1]
med = [1, 0, 1, 0, 1, 0]
slow = [1, 0, 0, 1, 0, 0]

# fast_to_slow_rhythms = [med, slow, fast]
slow_to_slow_rhythms = [med, fast, slow]

### multi-timescale multi-material all-to-all
multi_multi_all_to_all_df = pd.DataFrame(columns=["brl heart", "blr heart", "brl chest", "blr chest", "brl blood", "blr blood"])
multi_multi_all_all_buf = "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/multi_multi_all_to_all.csv"

### multi-timescale multi-material 
multi_multi_one_to_one_df = pd.DataFrame(columns=["brl heart", "blr heart", "brl chest", "blr chest", "brl blood", "blr blood"])
multi_multi_one_to_one_buf = "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/multi_multi_one_to_one.csv"

def run():
    brl_matlist = {
        0: bucket.Bucket(subreservoir_size, 0, 0),
        1: ring.DelayLine(subreservoir_size, 0, 0),
        2: lattice.MagLattice(subreservoir_size, 0, 0)
    }
    # blr_matlist =  {
    #     0: bucket.Bucket(subreservoir_size, 0, 0),
    #     1: lattice.MagLattice(subreservoir_size, 0, 0),
    #     2: ring.DelayLine(subreservoir_size, 0, 0)
    # }
    lrb_matlist = {
    0: lattice.MagLattice(subreservoir_size, 0, 0),
    1: ring.DelayLine(subreservoir_size, 0, 0),
    2: bucket.Bucket(subreservoir_size, 0, 0)
    }

    input_mask = np.block([[np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 2))],
                        [np.zeros((subreservoir_size, 1)), np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 1))],
                        [np.zeros((subreservoir_size, 2)), np.ones((subreservoir_size, 1))]])

    for i in range(TRuns):
        print(i)
        lrb_multi_timescale_oto = create_esns.create_esn_multi_timescale(lrb_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=slow_to_slow_rhythms)
        lrb_multi_timescale_oto.set_input_weights(lrb_multi_timescale_oto.Wu * input_mask)
        lrb_multi_results = sleep.run_benchmark(lrb_multi_timescale_oto, data_lengths)
        multi_multi_one_to_one_df.at[i, "lrb heart"] = lrb_multi_results[0]
        multi_multi_one_to_one_df.at[i, "lrb chest"] = lrb_multi_results[1]
        multi_multi_one_to_one_df.at[i, "lrb blood"] = lrb_multi_results[2]

        brl_multi_timescale_oto = create_esns.create_esn_multi_timescale(brl_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=slow_to_slow_rhythms)
        brl_multi_timescale_oto.set_input_weights(brl_multi_timescale_oto.Wu * input_mask)
        brl_multi_results = sleep.run_benchmark(brl_multi_timescale_oto, data_lengths)
        multi_multi_one_to_one_df.at[i, "brl heart"] = brl_multi_results[0]
        multi_multi_one_to_one_df.at[i, "brl chest"] = brl_multi_results[1]
        multi_multi_one_to_one_df.at[i, "brl blood"] = brl_multi_results[2]

        lrb_multi_timescale_ata = create_esns.create_esn_multi_timescale(lrb_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=slow_to_slow_rhythms)
        lrb_multi_ata_results = sleep.run_benchmark(lrb_multi_timescale_ata, data_lengths)
        multi_multi_all_to_all_df.at[i, "lrb heart"] = lrb_multi_ata_results[0]
        multi_multi_all_to_all_df.at[i, "lrb chest"] = lrb_multi_ata_results[1]
        multi_multi_all_to_all_df.at[i, "lrb blood"] = lrb_multi_ata_results[2]

        brl_multi_timescale_ata = create_esns.create_esn_multi_timescale(brl_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=slow_to_slow_rhythms)
        brl_multi_ata_results = sleep.run_benchmark(brl_multi_timescale_ata, data_lengths)
        multi_multi_all_to_all_df.at[i, "brl heart"] = brl_multi_ata_results[0]
        multi_multi_all_to_all_df.at[i, "brl chest"] = brl_multi_ata_results[1]
        multi_multi_all_to_all_df.at[i, "brl blood"] = brl_multi_ata_results[2]
    multi_multi_all_to_all_df.to_csv(multi_multi_all_all_buf)
    multi_multi_one_to_one_df.to_csv(multi_multi_one_to_one_buf)
    return

run()