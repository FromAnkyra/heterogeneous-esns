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

TWashout = 100
TTest = 200
TTrain = 800

data_lengths = (TWashout, TTrain, TTest)

TRuns = 50 
fullsize = 64*3
subreservoir_size = 64

def setup_esn(i, mat, size=fullsize):
    W = mat.generate_W(i, debug=False)
    esn = nymph.NymphESN(3, size, 1, density=0.1, seed=i, svd_dv=1)
    esn.set_weights(W)
    return esn
esn_df = pd.DataFrame(columns=["bucket", "ring", "lattice"])
esn_buf = "/home/cw1647/phd/het_reservoir_experiments/mso/results/single_reservoir.csv"

single_single_all_to_all_df = pd.DataFrame(columns=["bucket", "ring", "lattice"])
single_single_all_to_all_buf = "/home/cw1647/phd/het_reservoir_experiments/mso/results/single_single_all_to_all.csv"

single_single_one_to_one_df = pd.DataFrame(columns=["bucket", "ring", "lattice"])
single_single_one_to_one_buf = "/home/cw1647/phd/het_reservoir_experiments/mso/results/single_single_one_to_one.csv"

def run_experiment():
    input_mask = np.block([[np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 2))],
                  [np.zeros((subreservoir_size, 1)), np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 1))],
                  [np.zeros((subreservoir_size, 2)), np.ones((subreservoir_size, 1))]])
    for i in range(TRuns):
        print(i)
        bucketesn = setup_esn(i, bucket.Bucket(fullsize, 0, 0))
        esn_df.at[i, "bucket"] = mso.run_MSO_multi_input(bucketesn, MSO_list, resolution_4, data_lengths)[1]
        latticeesn = setup_esn(i, lattice.MagLattice(64*4, 0, 0), size=64*4) # has to be this size for Square
        esn_df.at[i, "lattice"] = mso.run_MSO_multi_input(latticeesn, MSO_list, resolution_4, data_lengths) [1]
        ringesn = setup_esn(i, ring.DelayLine(fullsize, 0, 0)) 
        esn_df.at[i, "ring"] = mso.run_MSO_multi_input(ringesn, MSO_list, resolution_4, data_lengths) [1]

        bucketata = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True, K=3)
        single_single_all_to_all_df.at[i, "bucket"] = mso.run_MSO_multi_input(bucketata, MSO_list, resolution_4, data_lengths)[1]
        ringata = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True, K=3)
        single_single_all_to_all_df.at[i, "ring"] = mso.run_MSO_multi_input(ringata, MSO_list, resolution_4, data_lengths)[1]
        latticeata = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True, K=3)
        single_single_all_to_all_df.at[i, "lattice"] = mso.run_MSO_multi_input(latticeata, MSO_list, resolution_4, data_lengths)[1]

        # bucket_oto = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True, K=3)
        # single_single_one_to_one_df.at[i, "bucket"] = mso.run_MSO_multi_input(bucket_oto, MSO_list, resolution_4, data_lengths, error="nrmse", input_mapping=True, input_mask=input_mask)[1]
        # ring_oto = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True, K=3)
        # single_single_one_to_one_df.at[i, "ring"] = mso.run_MSO_multi_input(ring_oto, MSO_list, resolution_4, data_lengths, error="nrmse", input_mapping=True, input_mask=input_mask)[1]
        # lattice_oto = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True, K=3)
        # single_single_one_to_one_df.at[i, "lattice"] = mso.run_MSO_multi_input(lattice_oto, MSO_list, resolution_4, data_lengths, error="nrmse", input_mapping=True, input_mask=input_mask)[1]
        
    esn_df.to_csv(esn_buf)
    single_single_all_to_all_df.to_csv(single_single_all_to_all_buf)
    # single_single_one_to_one_df.to_csv(single_single_one_to_one_buf)
    return


run_experiment()