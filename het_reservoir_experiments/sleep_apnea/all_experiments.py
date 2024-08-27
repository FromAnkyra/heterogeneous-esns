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

density = 0.1
db = 0.025

### ESN setup


def setup_esn(density, i):
    return nymph.NymphESN(3, fullsize, 1, density=density, seed=i, svd_dv=1)

esn_df = pd.DataFrame(columns=["heart", "chest", "blood"])
esn_buf = "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/esn_results.csv"
### Restricted ESN setup

def setup_resn(DW, DB, i):
    W = rmatrix.create_restricted_esn_weights(fullsize, subreservoir_size, 3, DW, DB)
    restricted = nymph.NymphESN(3, fullsize, 1, density=DW, seed=i, svd_dv=1)
    restricted.set_weights(W)
    return restricted

resn_df = pd.DataFrame(columns=["heart", "chest", "blood"])
resn_buf = "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/resn_results.csv"

### multi-timescale single material all-to-all input

multi_timescale_all_to_all_df = pd.DataFrame(columns=["bucket heart", "ring heart", "lattice heart", "bucket chest", "ring chest", "lattice chest", "bucket blood", "ring blood", "lattice blood"])
multi_timescale_all_to_all_buf = "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/multi_timescale_all_to_all.csv"

fast = [1, 1, 1, 1, 1, 1]
med = [1, 0, 1, 0, 1, 0]
slow = [1, 0, 0, 1, 0, 0]

slow_to_slow_rhythms = [med, fast, slow]

### multi-timescale single material one-to-one input

multi_timescale_one_to_one_df = pd.DataFrame(columns=["bucket heart", "ring heart", "lattice heart", "bucket chest", "ring chest", "lattice chest", "bucket blood", "ring blood", "lattice blood"])
multi_timescale_one_to_one_buf = "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/multi_timescale_one_to_one.csv"

### single timescale multi-material all-to-all input

multi_material_all_to_all_df = pd.DataFrame(columns=["brl heart", "lrb heart", "brl chest", "lrb chest", "brl blood", "lrb blood"])
multi_material_all_to_all_buf = "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/multi_material_all_to_all.csv"

### single timescale multi-material one-to-one input

multi_material_one_to_one_df = pd.DataFrame(columns=["brl heart", "lrb heart", "brl chest", "lrb chest", "brl blood", "lrb blood"])
multi_material_one_to_one_buf = "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/multi_material_one_to_one.csv"

### multi-timescale multi-material 
# multi_multi_df = pd.DataFrame(columns=["brl heart", "blr heart", "brl chest", "blr chest", "brl blood", "blr blood"])
# multi_multi_buf = "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/multi_multi.csv"

### multi-timescale multi-material all-to-all
multi_multi_all_to_all_df = pd.DataFrame(columns=["brl heart", "blr heart", "brl chest", "blr chest", "brl blood", "blr blood"])
multi_multi_all_all_buf = "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/multi_multi_all_to_all.csv"

### multi-timescale multi-material 
multi_multi_one_to_one_df = pd.DataFrame(columns=["brl heart", "blr heart", "brl chest", "blr chest", "brl blood", "blr blood"])
multi_multi_one_to_one_buf = "/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/multi_multi_one_to_one.csv"
### run everything!
def run(dw, db):
    brl_matlist = {
        0: bucket.Bucket(subreservoir_size, 0, 0),
        1: ring.DelayLine(subreservoir_size, 0, 0),
        2: lattice.MagLattice(subreservoir_size, 0, 0)
    }
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

        # esn = setup_esn(dw, i)
        # esn_results = sleep.run_benchmark(esn, data_lengths)
        # esn_df.at[i, "heart"] = esn_results[0]
        # esn_df.at[i, "chest"] = esn_results[1]
        # esn_df.at[i, "blood"] = esn_results[2]
        
        # resn = setup_resn(dw, db, i)
        # resn_results = sleep.run_benchmark(resn, data_lengths)
        # resn_df.at[i, "heart"] = resn_results[0]
        # resn_df.at[i, "chest"] = resn_results[1]
        # resn_df.at[i, "blood"] = resn_results[2]

        # multi-timescale all-to-all
        # bucketesn = create_esns.create_esn_multi_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=slow_to_slow_rhythms)
        # bucket_results = sleep.run_benchmark(bucketesn, data_lengths)
        # multi_timescale_all_to_all_df.at[i, "bucket heart"] = bucket_results[0]
        # multi_timescale_all_to_all_df.at[i, "bucket chest"] = bucket_results[1]
        # multi_timescale_all_to_all_df.at[i, "bucket blood"] = bucket_results[2]
        
        # ringesn = create_esns.create_esn_multi_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=slow_to_slow_rhythms)
        # ring_results = sleep.run_benchmark(ringesn, data_lengths)
        # multi_timescale_all_to_all_df.at[i, "ring heart"] = ring_results[0]
        # multi_timescale_all_to_all_df.at[i, "ring chest"] = ring_results[1]
        # multi_timescale_all_to_all_df.at[i, "ring blood"] = ring_results[2]
    
        
        # latticeesn = create_esns.create_esn_multi_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=slow_to_slow_rhythms)
        # lattice_results = sleep.run_benchmark(latticeesn, data_lengths)
        # multi_timescale_all_to_all_df.at[i, "lattice heart"] = lattice_results[0]
        # multi_timescale_all_to_all_df.at[i, "lattice chest"] = lattice_results[1]
        # multi_timescale_all_to_all_df.at[i, "lattice blood"] = lattice_results[2]
        
        # multi-material all-to-all  
        brl = create_esns.create_esn_single_timescale(brl_matlist, fullsize, i, normalise_svd=True, K=3)
        brl_results = sleep.run_benchmark(brl, data_lengths)
        multi_material_all_to_all_df.at[i, "brl heart"] = brl_results[0]
        multi_material_all_to_all_df.at[i, "brl chest"] = brl_results[1]
        multi_material_all_to_all_df.at[i, "brl blood"] = brl_results[2]

        lrb = create_esns.create_esn_single_timescale(lrb_matlist, fullsize, i, normalise_svd=True, K=3)
        lrb_results = sleep.run_benchmark(lrb, data_lengths)
        multi_material_all_to_all_df.at[i, "lrb heart"] = lrb_results[0]
        multi_material_all_to_all_df.at[i, "lrb chest"] = lrb_results[1]
        multi_material_all_to_all_df.at[i, "lrb blood"] = lrb_results[2]

        #multi-timescale one-to-one

        # bucket_oto = create_esns.create_esn_multi_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=slow_to_slow_rhythms)
        # bucket_oto.set_input_weights(bucket_oto.Wu * input_mask)
        # bucket_oto_results = sleep.run_benchmark(bucket_oto, data_lengths)
        # multi_timescale_one_to_one_df.at[i, "bucket heart"] = bucket_oto_results[0]
        # multi_timescale_one_to_one_df.at[i, "bucket chest"] = bucket_oto_results[1]
        # multi_timescale_one_to_one_df.at[i, "bucket blood"] = bucket_oto_results[2]
        
        # ring_oto = create_esns.create_esn_multi_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=slow_to_slow_rhythms)
        # ring_oto.set_input_weights(ring_oto.Wu * input_mask)
        # ring_oto_results = sleep.run_benchmark(ring_oto, data_lengths)
        # multi_timescale_one_to_one_df.at[i, "ring heart"] = ring_oto_results[0]
        # multi_timescale_one_to_one_df.at[i, "ring chest"] = ring_oto_results[1]
        # multi_timescale_one_to_one_df.at[i, "ring blood"] = ring_oto_results[2]


        # lattice_oto = create_esns.create_esn_multi_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=slow_to_slow_rhythms)
        # lattice_oto.set_input_weights(lattice_oto.Wu * input_mask)
        # lattice_oto_results = sleep.run_benchmark(lattice_oto, data_lengths)
        # multi_timescale_one_to_one_df.at[i, "lattice heart"] = lattice_oto_results[0]
        # multi_timescale_one_to_one_df.at[i, "lattice chest"] = lattice_oto_results[1]
        # multi_timescale_one_to_one_df.at[i, "lattice blood"] = lattice_oto_results[2]

        #multi-material one-to-one
        brl_oto = create_esns.create_esn_single_timescale(brl_matlist, fullsize, i, normalise_svd=True, K=3)
        brl_oto.set_input_weights(brl_oto.Wu * input_mask)        
        brl_oto_results = sleep.run_benchmark(brl_oto, data_lengths)
        multi_material_one_to_one_df.at[i, "brl heart"] = brl_oto_results[0]
        multi_material_one_to_one_df.at[i, "brl chest"] = brl_oto_results[1]
        multi_material_one_to_one_df.at[i, "brl blood"] = brl_oto_results[2]

        lrb_oto = create_esns.create_esn_single_timescale(lrb_matlist, fullsize, i, normalise_svd=True, K=3)
        lrb_oto.set_input_weights(lrb_oto.Wu * input_mask)
        lrb_oto_results = sleep.run_benchmark(lrb_oto, data_lengths)
        multi_material_one_to_one_df.at[i, "lrb heart"] = lrb_oto_results[0]
        multi_material_one_to_one_df.at[i, "lrb chest"] = lrb_oto_results[1]
        multi_material_one_to_one_df.at[i, "lrb blood"] = lrb_oto_results[2]

        #multi-timescale multi-material one-to-one
        # blr_multi_timescale_oto = create_esns.create_esn_multi_timescale(blr_matlist, fullsize, i, normalise_svd=True, K=3)
        # blr_multi_timescale_oto.set_input_weights(blr_multi_timescale_oto.Wu * input_mask)
        # blr_multi_results = sleep.run_benchmark(blr_multi_timescale_oto, data_lengths)
        # multi_multi_one_to_one_df.at[i, "blr heart"] = blr_multi_results[0]
        # multi_multi_one_to_one_df.at[i, "blr chest"] = blr_multi_results[1]
        # multi_multi_one_to_one_df.at[i, "blr blood"] = blr_multi_results[2]

        # brl_multi_timescale_oto = create_esns.create_esn_multi_timescale(brl_matlist, fullsize, i, normalise_svd=True, K=3)
        # brl_multi_timescale_oto.set_input_weights(brl_multi_timescale_oto.Wu * input_mask)
        # brl_multi_results = sleep.run_benchmark(brl_multi_timescale_oto, data_lengths)
        # multi_multi_one_to_one_df.at[i, "brl heart"] = brl_multi_results[0]
        # multi_multi_one_to_one_df.at[i, "brl chest"] = brl_multi_results[1]
        # multi_multi_one_to_one_df.at[i, "brl blood"] = brl_multi_results[2]

        #multi-multi all-to-all
        # blr_multi_timescale_ata = create_esns.create_esn_multi_timescale(blr_matlist, fullsize, i, normalise_svd=True, K=3)
        # blr_multi_ata_results = sleep.run_benchmark(blr_multi_timescale_ata, data_lengths)
        # multi_multi_all_to_all_df.at[i, "blr heart"] = blr_multi_ata_results[0]
        # multi_multi_all_to_all_df.at[i, "blr chest"] = blr_multi_ata_results[1]
        # multi_multi_all_to_all_df.at[i, "blr blood"] = blr_multi_ata_results[2]

        # brl_multi_timescale_ata = create_esns.create_esn_multi_timescale(brl_matlist, fullsize, i, normalise_svd=True, K=3)
        # brl_multi_ata_results = sleep.run_benchmark(brl_multi_timescale_ata, data_lengths)
        # multi_multi_all_to_all_df.at[i, "brl heart"] = brl_multi_ata_results[0]
        # multi_multi_all_to_all_df.at[i, "brl chest"] = brl_multi_ata_results[1]
        # multi_multi_all_to_all_df.at[i, "brl blood"] = brl_multi_ata_results[2]
    
    # esn_df.to_csv(esn_buf)
    # resn_df.to_csv(resn_buf)
    # multi_timescale_all_to_all_df.to_csv(multi_timescale_all_to_all_buf)
    # multi_timescale_one_to_one_df.to_csv(multi_timescale_one_to_one_buf)
    multi_material_all_to_all_df.to_csv(multi_material_all_to_all_buf)
    multi_material_one_to_one_df.to_csv(multi_material_one_to_one_buf)
    # multi_multi_all_to_all_df.to_csv(multi_multi_all_all_buf)
    # multi_multi_one_to_one_df.to_csv(multi_multi_one_to_one_buf)
    # multi_multi_df.to_csv(multi_multi_buf)

    return

run(density, db)