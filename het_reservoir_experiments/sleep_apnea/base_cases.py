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

def setup_esn(i, mat, size=fullsize):
    W = mat.generate_W(i, debug=False)
    esn = nymph.NymphESN(3, size, 1, density=0.1, seed=i, svd_dv=1)
    esn.set_weights(W)
    # print(f"{esn.K=}")
    return esn
esn_df = pd.DataFrame(columns=["bucket heart", "ring heart", "lattice heart", "bucket chest", "ring chest", "lattice chest", "bucket blood", "ring blood", "lattice blood"])
esn_buf = "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/single_reservoir.csv"

single_single_all_to_all_df = pd.DataFrame(columns=["bucket heart", "ring heart", "lattice heart", "bucket chest", "ring chest", "lattice chest", "bucket blood", "ring blood", "lattice blood"])
single_single_all_to_all_buf = "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/single_single_all_to_all.csv"

single_single_one_to_one_df = pd.DataFrame(columns=["bucket heart", "ring heart", "lattice heart", "bucket chest", "ring chest", "lattice chest", "bucket blood", "ring blood", "lattice blood"])
single_single_one_to_one_buf = "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/single_single_one_to_one.csv"

def run_experiment():
    input_mask = np.block([[np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 2))],
                  [np.zeros((subreservoir_size, 1)), np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 1))],
                  [np.zeros((subreservoir_size, 2)), np.ones((subreservoir_size, 1))]])
    for i in range(TRuns):
        print(i)
        bucketesn = setup_esn(i, bucket.Bucket(fullsize, 0, 0))
        bucketesn_results = sleep.run_benchmark(bucketesn, data_lengths)
        esn_df.at[i, "bucket heart"] = bucketesn_results[0]
        esn_df.at[i, "bucket chest"] = bucketesn_results[1]
        esn_df.at[i, "bucket blood"] = bucketesn_results[2]
        latticeesn = setup_esn(i, lattice.MagLattice(64*4, 0, 0), size=64*4) # has to be this size for Square
        latticeesn_results = sleep.run_benchmark(latticeesn, data_lengths)
        esn_df.at[i, "lattice heart"] = latticeesn_results[0]
        esn_df.at[i, "lattice chest"] = latticeesn_results[1]
        esn_df.at[i, "lattice blood"] = latticeesn_results[2]
        ringesn = setup_esn(i, ring.DelayLine(fullsize, 0, 0)) 
        ringesn_results = sleep.run_benchmark(ringesn, data_lengths)
        esn_df.at[i, "ring heart"] = ringesn_results[0]
        esn_df.at[i, "ring chest"] = ringesn_results[1]
        esn_df.at[i, "ring blood"] = ringesn_results[2]
        
        bucketata = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True, K=3)
        bucketata_results = sleep.run_benchmark(bucketata, data_lengths)
        single_single_all_to_all_df.at[i, "bucket heart"] = bucketata_results[0]
        single_single_all_to_all_df.at[i, "bucket chest"] = bucketata_results[1]
        single_single_all_to_all_df.at[i, "bucket blood"] = bucketata_results[2]        
        latticeata = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True, K=3)
        latticeata_results = sleep.run_benchmark(latticeata, data_lengths)
        single_single_all_to_all_df.at[i, "lattice heart"] = latticeata_results[0]
        single_single_all_to_all_df.at[i, "lattice chest"] = latticeata_results[1]
        single_single_all_to_all_df.at[i, "lattice blood"] = latticeata_results[2]
        ringata = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True, K=3)        
        ringata_results = sleep.run_benchmark(ringata, data_lengths)
        single_single_all_to_all_df.at[i, "ring heart"] = ringata_results[0]
        single_single_all_to_all_df.at[i, "ring chest"] = ringata_results[1]
        single_single_all_to_all_df.at[i, "ring blood"] = ringata_results[2]

        bucket_oto = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True, K=3)
        bucket_oto.set_input_weights(bucket_oto.Wu * input_mask)
        bucket_oto_results = sleep.run_benchmark(bucket_oto, data_lengths)
        single_single_one_to_one_df.at[i, "bucket heart"] = bucket_oto_results[0]
        single_single_one_to_one_df.at[i, "bucket chest"] = bucket_oto_results[1]
        single_single_one_to_one_df.at[i, "bucket blood"] = bucket_oto_results[2]
        ring_oto = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True, K=3)
        ring_oto.set_input_weights(ring_oto.Wu * input_mask)
        ring_oto_results = sleep.run_benchmark(ring_oto, data_lengths)
        single_single_one_to_one_df.at[i, "ring heart"] = ring_oto_results[0]
        single_single_one_to_one_df.at[i, "ring chest"] = ring_oto_results[1]
        single_single_one_to_one_df.at[i, "ring blood"] = ring_oto_results[2]
        lattice_oto = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True, K=3)
        lattice_oto.set_input_weights(lattice_oto.Wu * input_mask)
        lattice_oto_results = sleep.run_benchmark(lattice_oto, data_lengths)
        single_single_one_to_one_df.at[i, "lattice heart"] = lattice_oto_results[0]
        single_single_one_to_one_df.at[i, "lattice chest"] = lattice_oto_results[1]
        single_single_one_to_one_df.at[i, "lattice blood"] = lattice_oto_results[2]
        
    esn_df.to_csv(esn_buf)
    single_single_all_to_all_df.to_csv(single_single_all_to_all_buf)
    single_single_one_to_one_df.to_csv(single_single_one_to_one_buf)
    return


run_experiment()