import numpy as np
import pandas as pd
import sys
import fakemat_experiments.create_esns as create_esns


fullsize = 64*3
subreservoir_size = 64

def experiment(seed):
    print("start")
    print(f"{seed=}")
    bucketesn = create_esns.create_esn_single_timescale(create_esns.bucket_matlist, fullsize, seed, normalise_svd=True, debug=True)  
    delaylineesn = create_esns.create_esn_single_timescale(create_esns.delayline_matlist, fullsize, seed, normalise_svd=True, debug=True)
    maglatticeesn = create_esns.create_esn_single_timescale(create_esns.maglattice_matlist, fullsize, seed, normalise_svd=True, debug=True)
    print("done!")
    return

experiment(1)