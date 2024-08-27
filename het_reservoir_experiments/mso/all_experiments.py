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
    return nymph.NymphESN(3, fullsize, 1, density=density, seed=i, svd_dv=1)

esn_df = pd.DataFrame(columns=["test"])
esn_buf = "/home/cw1647/phd/het_reservoir_experiments/mso/results/esn_results.csv"
### Restricted ESN setup

def setup_resn(DW, DB, i):
    W = rmatrix.create_restricted_esn_weights(fullsize, subreservoir_size, 3, DW, DB)
    restricted = nymph.NymphESN(3, fullsize, 1, density=DW, seed=i, svd_dv=1)
    restricted.set_weights(W)
    return restricted

resn_df = pd.DataFrame(columns=["test"])
resn_buf = "/home/cw1647/phd/het_reservoir_experiments/mso/results/resn_results.csv"

### multi-timescale single material all-to-all input

multi_timescale_all_to_all_df = pd.DataFrame(columns=["bucket", "ring", "lattice"])
multi_timescale_all_to_all_buf = "/home/cw1647/phd/het_reservoir_experiments/mso/results/multi_timescale_all_to_all.csv"

### multi-timescale single material one-to-one input

multi_timescale_one_to_one_df = pd.DataFrame(columns=["bucket", "ring", "lattice"])
multi_timescale_one_to_one_buf = "/home/cw1647/phd/het_reservoir_experiments/mso/results/multi_timescale_one_to_one.csv"

### single timescale multi-material all-to-all input

multi_material_all_to_all_df = pd.DataFrame(columns=["rlb", "rbl"])
multi_material_all_to_all_buf = "/home/cw1647/phd/het_reservoir_experiments/mso/results/multi_material_all_to_all.csv"

### single timescale multi-material one-to-one input

multi_material_one_to_one_df = pd.DataFrame(columns=["rlb", "rbl"])
multi_material_one_to_one_buf = "/home/cw1647/phd/het_reservoir_experiments/mso/results/multi_material_one_to_one.csv"

### multi-timescale multi-material 
multi_multi_df = pd.DataFrame(columns=["rlb", "rbl"])
multi_multi_buf = "/home/cw1647/phd/het_reservoir_experiments/mso/results/multi_multi_one_to_one.csv"

multi_multi_all_to_all_df = pd.DataFrame(columns=["rlb", "rbl"])
multi_multi_all_to_all_buf = "/home/cw1647/phd/het_reservoir_experiments/mso/results/multi_multi_all_to_all.csv"


# timescales
fast = [1, 1, 1, 1, 1, 1]
med = [1, 0, 1, 0, 1, 0]
slow = [1, 0, 0, 1, 0, 0]

fast_to_fast_rhythms = [slow, med, fast]

### run everything!
def run(dw, db):
    
    rlb_matlist = {
        0: ring.DelayLine(subreservoir_size, 0, 0),
        1: lattice.MagLattice(subreservoir_size, 0, 0),
        2: bucket.Bucket(subreservoir_size, 0, 0)
    }
    rbl_matlist =  {
        0: ring.DelayLine(subreservoir_size, 0, 0),
        1: bucket.Bucket(subreservoir_size, 0, 0),
        2: lattice.MagLattice(subreservoir_size, 0, 0)
    }

    input_mask = np.block([[np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 2))],
                      [np.zeros((subreservoir_size, 1)), np.ones((subreservoir_size, 1)), np.zeros((subreservoir_size, 1))],
                      [np.zeros((subreservoir_size, 2)), np.ones((subreservoir_size, 1))]])

    for i in range(TRuns):
        print(i)

        esn = setup_esn(dw, i)
        esn_df.at[i, "test"] = mso.run_MSO_multi_input(esn, MSO_list, resolution_4, data_lengths)[1]
        
        resn = setup_resn(dw, db, i)
        resn_df.at[i, "test"] = mso.run_MSO_multi_input(resn, MSO_list, resolution_4, data_lengths)[1]
        
        # multi-timescale all-to-all
        bucketesn = create_esns.create_esn_multi_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=fast_to_fast_rhythms)
        multi_timescale_all_to_all_df.at[i, "bucket"] =mso.run_MSO_multi_input(bucketesn, MSO_list, resolution_4, data_lengths)[1]
        ringesn = create_esns.create_esn_multi_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=fast_to_fast_rhythms)
        multi_timescale_all_to_all_df.at[i, "ring"] = mso.run_MSO_multi_input(ringesn, MSO_list, resolution_4, data_lengths)[1]
        latticeesn = create_esns.create_esn_multi_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=fast_to_fast_rhythms)
        multi_timescale_all_to_all_df.at[i, "lattice"] = mso.run_MSO_multi_input(latticeesn, MSO_list, resolution_4, data_lengths)[1]

        # multi-material all-to-all  
        rlb = create_esns.create_esn_single_timescale(rlb_matlist, fullsize, i, normalise_svd=True, K=3)
        multi_material_all_to_all_df.at[i, "rlb"] = mso.run_MSO_multi_input(rlb, MSO_list, resolution_4, data_lengths)[1]
        rbl = create_esns.create_esn_single_timescale(rbl_matlist, fullsize, i, normalise_svd=True, K=3)
        multi_material_all_to_all_df.at[i, "rbl"] = mso.run_MSO_multi_input(rbl, MSO_list, resolution_4, data_lengths)[1]

        #multi-timescale one-to-one

        # bucket_oto = create_esns.create_esn_multi_timescale(create_esns.bucket_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=fast_to_fast_rhythms)
        # multi_timescale_one_to_one_df.at[i, "bucket"] = mso.run_MSO_multi_input(bucket_oto, MSO_list, resolution_4, data_lengths, error="nrmse", input_mapping=True, input_mask=input_mask)[1]
        # ring_oto = create_esns.create_esn_multi_timescale(create_esns.delayline_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=fast_to_fast_rhythms)
        # multi_timescale_one_to_one_df.at[i, "ring"] = mso.run_MSO_multi_input(ring_oto, MSO_list, resolution_4, data_lengths, error="nrmse", input_mapping=True, input_mask=input_mask)[1]
        # lattice_oto = create_esns.create_esn_multi_timescale(create_esns.maglattice_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=fast_to_fast_rhythms)
        # multi_timescale_one_to_one_df.at[i, "lattice"] = mso.run_MSO_multi_input(lattice_oto, MSO_list, resolution_4, data_lengths, input_mapping=True, input_mask=input_mask)[1]

        #multi-material one-to-one
        # rlb_oto = create_esns.create_esn_single_timescale(rlb_matlist, fullsize, i, normalise_svd=True, K=3)
        # multi_material_one_to_one_df.at[i, "rlb"] = mso.run_MSO_multi_input(rlb_oto, MSO_list, resolution_4, data_lengths, error="nrmse", input_mapping=True, input_mask=input_mask)[1]
        # rbl_oto = create_esns.create_esn_single_timescale(rbl_matlist, fullsize, i, normalise_svd=True, K=3)
        # multi_material_one_to_one_df.at[i, "rbl"] = mso.run_MSO_multi_input(rbl_oto, MSO_list, resolution_4, data_lengths, error="nrmse", input_mapping=True, input_mask=input_mask)[1]

        #multi-timescale multi-material one-to-one
        # rlb_multi_timescale = create_esns.create_esn_multi_timescale(rlb_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=fast_to_fast_rhythms)
        # multi_multi_df.at[i, "rlb"] = mso.run_MSO_multi_input(rlb_multi_timescale, MSO_list, resolution_4, data_lengths, error="nrmse", input_mapping=True, input_mask=input_mask)[1]
        # rbl_multi_timescale = create_esns.create_esn_multi_timescale(rbl_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=fast_to_fast_rhythms)
        # multi_multi_df.at[i, "rbl"] = mso.run_MSO_multi_input(rbl_multi_timescale, MSO_list, resolution_4, data_lengths, error="nrmse", input_mapping=True, input_mask=input_mask)[1]

        rlb_multi_timescale_all_to_all = create_esns.create_esn_multi_timescale(rlb_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=fast_to_fast_rhythms)
        multi_multi_all_to_all_df.at[i, "rlb"] = mso.run_MSO_multi_input(rlb_multi_timescale_all_to_all, MSO_list, resolution_4, data_lengths)[1]
        rbl_multi_timescale_all_to_all = create_esns.create_esn_multi_timescale(rbl_matlist, fullsize, i, normalise_svd=True, K=3, rhythms=fast_to_fast_rhythms)
        multi_multi_all_to_all_df.at[i, "rbl"] = mso.run_MSO_multi_input(rbl_multi_timescale_all_to_all, MSO_list, resolution_4, data_lengths)[1]
    
    esn_df.to_csv(esn_buf)
    resn_df.to_csv(resn_buf)
    multi_timescale_all_to_all_df.to_csv(multi_timescale_all_to_all_buf)
    # multi_timescale_one_to_one_df.to_csv(multi_timescale_one_to_one_buf)
    multi_material_all_to_all_df.to_csv(multi_material_all_to_all_buf)
    # multi_material_one_to_one_df.to_csv(multi_material_one_to_one_buf)
    # multi_multi_df.to_csv(multi_multi_buf)
    multi_multi_all_to_all_df.to_csv(multi_multi_all_to_all_buf)
    return

run(density, db)