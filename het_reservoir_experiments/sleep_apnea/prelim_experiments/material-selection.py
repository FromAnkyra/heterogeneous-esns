import numpy as np 
import benchmarks.sleep_apnea as sleep
import pandas as pd
import fakemat_experiments.create_esns as create_esns
import fakemat_experiments.create_esns as create_esns
import fakemat_experiments.bucket as bucket
import fakemat_experiments.delayline as ring
import fakemat_experiments.maglattice as lattice

TStart = 22000
TWashout = 1000
TTrain = 3000
TTest = 1000

data_lengths = (TStart, TWashout, TTrain, TTest)

sub_size = 64
full_size = 64*3
TRuns = 50

input_mask = np.block([[np.ones((sub_size, 1)), np.zeros((sub_size, 2))],
                        [np.zeros((sub_size, 1)), np.ones((sub_size, 1)), np.zeros((sub_size, 1))],
                        [np.zeros((sub_size, 2)), np.ones((sub_size, 1))]])

rlb_matlist = {
    0: ring.DelayLine(sub_size, 0, 0),
    1: lattice.MagLattice(sub_size, 0, 0),
    2: bucket.Bucket(sub_size, 0, 0)
}

lrb_matlist = {
    0: lattice.MagLattice(sub_size, 0, 0),
    1: ring.DelayLine(sub_size, 0, 0),
    2: bucket.Bucket(sub_size, 0, 0)
}

brl_matlist = {
    0: bucket.Bucket(sub_size, 0, 0),
    1: ring.DelayLine(sub_size, 0, 0),
    2: lattice.MagLattice(sub_size, 0, 0)
}

blr_matlist = {
    0: bucket.Bucket(sub_size, 0, 0),
    1: lattice.MagLattice(sub_size, 0, 0),
    2: ring.DelayLine(sub_size, 0, 0)
}

rbl_matlist = {
    0: ring.DelayLine(sub_size, 0, 0),
    1: bucket.Bucket(sub_size, 0, 0),
    2: lattice.MagLattice(sub_size, 0, 0)
}

lbr_matlist = {
    0: lattice.MagLattice(sub_size, 0, 0),
    1: bucket.Bucket(sub_size, 0, 0),
    2: ring.DelayLine(sub_size, 0, 0)
}

def run_experiment(rhythms, path):
    print(path)
    rlbdf = pd.DataFrame(columns=['heart', "chest", "blood"])
    lrbdf = pd.DataFrame(columns=['heart', "chest", "blood"])
    brldf = pd.DataFrame(columns=['heart', "chest", "blood"])
    blrdf = pd.DataFrame(columns=['heart', "chest", "blood"])
    rbldf = pd.DataFrame(columns=['heart', "chest", "blood"])
    lbrdf = pd.DataFrame(columns=['heart', "chest", "blood"])
    for i in range(TRuns):
        print(i)
        rlb = create_esns.create_esn_multi_timescale(rlb_matlist, full_size, i, normalise_svd=True, K=3, rhythms=rhythms)
        rlb.set_input_weights(rlb.Wu*input_mask)
        rlb_nrmse = sleep.run_benchmark(rlb, data_lengths)
        rlbdf.at[i, "heart"] = rlb_nrmse[0]
        rlbdf.at[i, "chest"] = rlb_nrmse[1]
        rlbdf.at[i, "blood"] = rlb_nrmse[2]

        lrb = create_esns.create_esn_multi_timescale(lrb_matlist, full_size, i, normalise_svd=True, K=3, rhythms=rhythms)
        lrb.set_input_weights(lrb.Wu*input_mask)
        lrb_nrmse = sleep.run_benchmark(lrb, data_lengths)
        lrbdf.at[i, "heart"] = lrb_nrmse[0]
        lrbdf.at[i, "chest"] = lrb_nrmse[1]
        lrbdf.at[i, "blood"] = lrb_nrmse[2]

        brl = create_esns.create_esn_multi_timescale(brl_matlist, full_size, i, normalise_svd=True, K=3, rhythms=rhythms)
        brl.set_input_weights(brl.Wu*input_mask)
        brl_nrmse = sleep.run_benchmark(brl, data_lengths)
        brldf.at[i, "heart"] = brl_nrmse[0]
        brldf.at[i, "chest"] = brl_nrmse[1]
        brldf.at[i, "blood"] = brl_nrmse[2]

        blr = create_esns.create_esn_multi_timescale(blr_matlist, full_size, i, normalise_svd=True, K=3, rhythms=rhythms)
        blr.set_input_weights(blr.Wu*input_mask)
        blr_nrmse = sleep.run_benchmark(blr, data_lengths)
        blrdf.at[i, "heart"] = blr_nrmse[0]
        blrdf.at[i, "chest"] = blr_nrmse[1]
        blrdf.at[i, "blood"] = blr_nrmse[2]

        rbl = create_esns.create_esn_multi_timescale(rbl_matlist, full_size, i, normalise_svd=True, K=3, rhythms=rhythms)
        rbl.set_input_weights(rbl.Wu*input_mask)
        rbl_nrmse = sleep.run_benchmark(rbl, data_lengths)
        rbldf.at[i, "heart"] = rbl_nrmse[0]
        rbldf.at[i, "chest"] = rbl_nrmse[1]
        rbldf.at[i, "blood"] = rbl_nrmse[2]

        lbr = create_esns.create_esn_multi_timescale(lbr_matlist, full_size, i, normalise_svd=True, K=3, rhythms=rhythms)
        lbr.set_input_weights(lbr.Wu*input_mask)
        lbr_nrmse = sleep.run_benchmark(lbr, data_lengths)
        lbrdf.at[i, "heart"] = lbr_nrmse[0]
        lbrdf.at[i, "chest"] = lbr_nrmse[1]
        lbrdf.at[i, "blood"] = lbr_nrmse[2]
    rlbdf.to_csv(f"{path}/rlb.csv")
    lrbdf.to_csv(f"{path}/lrb.csv")
    brldf.to_csv(f"{path}/brl.csv")
    blrdf.to_csv(f"{path}/blr.csv")
    rbldf.to_csv(f"{path}/rbl.csv")
    lbrdf.to_csv(f"{path}/lbr.csv")

    return

fast = [1, 1, 1, 1, 1, 1]
med = [1, 0, 1, 0, 1, 0]
slow = [1, 0, 0, 1, 0, 0]

fast_to_slow_rhythms = [med, slow, fast]
slow_to_slow_rhythms = [med, fast, slow]

run_experiment(fast_to_slow_rhythms, "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/fast-to-slow")
run_experiment(slow_to_slow_rhythms, "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/slow-to-slow")