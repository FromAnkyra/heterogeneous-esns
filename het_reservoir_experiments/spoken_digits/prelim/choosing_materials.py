import numpy as np
import NymphESN.nymphesn as nymph
import benchmarks.spoken_digits as digits
import pandas as pd
import sys
import fakemat_experiments.create_esns as create_esns
import fakemat_experiments.bucket as bucket
import fakemat_experiments.delayline as ring
import fakemat_experiments.maglattice as lattice

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


data =digits.create_data(["/home/cw1647/phd/benchmarks/spoken_digits/ti46/AllTestData/", "/home/cw1647/phd/benchmarks/spoken_digits/ti46/AllTrainData"])
vhat_digits = data["test digits"]


def run_experiment():
    print("start!")
    errordf = pd.DataFrame(columns=["rlb", "lrb", "brl", "blr", "rbl", "lbr"])
    rhythms = [[1, 0, 0, 1, 0, 0], [1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1]] # mso.one.value is the slowest changing, so the slowest reservoir corresponds to it
    for i in range(TRuns):
        print(i)
        rlb = create_esns.create_esn_multi_timescale(rlb_matlist, full_size, i, normalise_svd=True, K=3)
        _, rlb_results = digits.run_spoken_digits(rlb, data)
        errordf.at[i, "rlb"] = digits.word_error_rate(digits.results_to_digits(rlb_results), vhat_digits)

        lrb = create_esns.create_esn_multi_timescale(lrb_matlist, full_size, i, normalise_svd=True, K=3)
        _, lrb_results = digits.run_spoken_digits(lrb, data)
        errordf.at[i, "lrb"] = digits.word_error_rate(digits.results_to_digits(lrb_results), vhat_digits)

        brl = create_esns.create_esn_multi_timescale(brl_matlist, full_size, i, normalise_svd=True, K=3)
        _, brl_results = digits.run_spoken_digits(brl, data)
        errordf.at[i, "brl"] = digits.word_error_rate(digits.results_to_digits(brl_results), vhat_digits)

        blr = create_esns.create_esn_multi_timescale(blr_matlist, full_size, i, normalise_svd=True, K=3)
        _, blr_results = digits.run_spoken_digits(blr, data)
        errordf.at[i, "blr"] = digits.word_error_rate(digits.results_to_digits(blr_results), vhat_digits)

        rbl = create_esns.create_esn_multi_timescale(rbl_matlist, full_size, i, normalise_svd=True, K=3)
        _, rbl_results = digits.run_spoken_digits(rbl, data)
        errordf.at[i, "rbl"] = digits.word_error_rate(digits.results_to_digits(rbl_results), vhat_digits)

        lbr = create_esns.create_esn_multi_timescale(lbr_matlist, full_size, i, normalise_svd=True, K=3)
        _, lbr_results = digits.run_spoken_digits(lbr, data)
        errordf.at[i, "lbr"] = digits.word_error_rate(digits.results_to_digits(lbr_results), vhat_digits)
    errordf.to_csv("/home/cw1647/phd/het_reservoir_experiments/spoken_digits/prelim/results/material-selection.csv")
    print("end")
    return

run_experiment()