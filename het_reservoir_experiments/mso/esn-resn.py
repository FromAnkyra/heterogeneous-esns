import numpy as np
import benchmarks.mso as mso
import NymphESN.nymphesn as nymph
import NymphESN.restrictedmatrix as rmatrix
import pandas as pd
import sys
import fakemat_experiments.create_esns as create_esns
import fakemat_experiments.bucket as bucket
import fakemat_experiments.delayline as ring
import fakemat_experiments.maglattice as lattice

resolution_4 = np.linspace(0, 500, 4000)
MSO_three = mso.generate_MSO(resolution_4, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value])
MSO_three = MSO_three/6 # scale the input to be between [-0.5, 0.5]
MSO_list = [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value]

# MSO_two = mso.generate_MSO(resolution_4, [mso.MSO.one.value, mso.MSO.two.value])
# MSO_two = MSO_two/4
# MSO_four = mso.generate_MSO(resolution_4, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value, mso.MSO.four.value])
# MSO_four = MSO_four/8

TWashout = 100
TTest = 200
TTrain = 800

data_lengths = (TWashout, TTrain, TTest)

TRuns = 50
fullsize = 64*3
subreservoir_size = 64

density = 0.1
db = 0.025

### ESN setup

def setup_esn(density, i):
    return nymph.NymphESN(1, fullsize, 1, density=density, seed=i, svd_dv=0.8)

esn_df = pd.DataFrame(columns=["test"])
esn_buf = "/home/cw1647/phd/het_reservoir_experiments/mso/results/esn_results.csv"
### Restricted ESN setup

def setup_resn(DW, DB, i):
    W = rmatrix.create_restricted_esn_weights(fullsize, subreservoir_size, 3, DW, DB)
    restricted = nymph.NymphESN(1, fullsize, 1, density=DW, seed=i, svd_dv=0.8)
    restricted.set_weights(W)
    return restricted

resn_df = pd.DataFrame(columns=["test"])
resn_buf = "/home/cw1647/phd/het_reservoir_experiments/mso/results/resn_results.csv"

def run(dw, db):
    for i in range(TRuns):
        print(i)

        esn = setup_esn(dw, i)
        result=mso.run_MSO_rr(esn, MSO_three, data_lengths, error="nrmse")[1]
        esn_df.at[i, "test"] = result
        # print(f"{type(result)}")
        resn = setup_resn(dw, db, i)
        resn_df.at[i, "test"] = mso.run_MSO_rr(resn, MSO_three, data_lengths, error="nrmse")[1]
    esn_df.to_csv(esn_buf)
    resn_df.to_csv(resn_buf)
    return

run(density, db)