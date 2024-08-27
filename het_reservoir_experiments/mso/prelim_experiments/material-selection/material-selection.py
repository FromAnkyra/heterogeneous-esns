import numpy as np
import benchmarks.mso as mso
import pandas as pd
import sys
import fakemat_experiments.create_esns as create_esns
import fakemat_experiments.bucket as bucket
import fakemat_experiments.delayline as ring
import fakemat_experiments.maglattice as lattice


resolution_4 = np.linspace(0, 500, 4000)

MSO_list = [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value]
MSO_three = mso.generate_MSO(resolution_4, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value])
MSO_three = MSO_three/6


TWashout = 100
TTest = 200
TTrain = 800

data_lengths = (TWashout, TTrain, TTest)

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

def run_experiment():
    print("start!")
    errordf = pd.DataFrame(columns=["rlb", "lrb", "brl", "blr", "rbl", "lbr"])
    rhythms = [[1, 0, 0, 1, 0, 0], [1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1]] # mso.one.value is the slowest changing, so the slowest reservoir corresponds to it
    for i in range(TRuns):
        print(i)
        rlb = create_esns.create_esn_multi_timescale(rlb_matlist, full_size, i, normalise_svd=True, K=3)
        errordf.at[i, "rlb"] = mso.run_MSO_multi_input(rlb, MSO_list, resolution_4, data_lengths, input_mapping=True, input_mask=input_mask)[1]

        lrb = create_esns.create_esn_multi_timescale(lrb_matlist, full_size, i, normalise_svd=True, K=3)
        errordf.at[i, "lrb"] = mso.run_MSO_multi_input(lrb, MSO_list, resolution_4, data_lengths, input_mapping=True, input_mask=input_mask)[1]

        brl = create_esns.create_esn_multi_timescale(brl_matlist, full_size, i, normalise_svd=True, K=3)
        errordf.at[i, "brl"] = mso.run_MSO_multi_input(brl, MSO_list, resolution_4, data_lengths, input_mapping=True, input_mask=input_mask)[1]

        blr = create_esns.create_esn_multi_timescale(blr_matlist, full_size, i, normalise_svd=True, K=3)
        errordf.at[i, "blr"] = mso.run_MSO_multi_input(blr, MSO_list, resolution_4, data_lengths, input_mapping=True, input_mask=input_mask)[1]

        rbl = create_esns.create_esn_multi_timescale(rbl_matlist, full_size, i, normalise_svd=True, K=3)
        errordf.at[i, "rbl"] = mso.run_MSO_multi_input(rbl, MSO_list, resolution_4, data_lengths, input_mapping=True, input_mask=input_mask)[1]

        lbr = create_esns.create_esn_multi_timescale(lbr_matlist, full_size, i, normalise_svd=True, K=3)
        errordf.at[i, "lbr"] = mso.run_MSO_multi_input(lbr, MSO_list, resolution_4, data_lengths, input_mapping=True, input_mask=input_mask)[1]
    errordf.to_csv("/home/cw1647/phd/het_reservoir_experiments/mso/prelim_experiments/material-selection/results/material-selection.csv")
    print("end")
    return

run_experiment()