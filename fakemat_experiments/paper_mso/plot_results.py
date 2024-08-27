import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def mso_by_material(path):
    materials_pal = {
        'MSO*-2 single': 'blue',
        'MSO*-2 multi': 'orange',
        'MSO*-4 single': 'blue',
        'MSO*-4 multi': 'orange',
        'MSO*-8 single': 'blue',
        'MSO*-8 multi': 'orange',
    }
    sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})  
    filenames = [str(path) for path in Path(path).glob('*') if "csv" in str(path)]
    delaylinefig, delaylineax = plt.subplots(1, 1, figsize=(3* 3, 4)) 
    delaylinedf = pd.DataFrame(columns=["MSO*-2 single", "MSO*-2 multi", "MSO*-4 single", "MSO*-4 multi", "MSO*-8 single", "MSO*-8 multi"]) 

    latticefig, latticeax = plt.subplots(1, 1, figsize=(3* 3, 4))
    latticedf = pd.DataFrame(columns=["MSO*-2 single", "MSO*-2 multi", "MSO*-4 single", "MSO*-4 multi", "MSO*-8 single", "MSO*-8 multi"]) 

    bucketfig, bucketax = plt.subplots(1, 1, figsize=(3* 3, 4))
    bucketdf = pd.DataFrame(columns=["MSO*-2 single", "MSO*-2 multi", "MSO*-4 single", "MSO*-4 multi", "MSO*-8 single", "MSO*-8 multi"]) 
    
    for f in filenames:
        print(f)
        df = pd.read_csv(f)
        if "eight" in f:
            if "single" in f:
                delaylinedf["MSO*-8 single"] = np.log10(df.loc[:, "delayline"].values)
                latticedf["MSO*-8 single"] = np.log10(df.loc[:, "maglattice"].values)
                bucketdf["MSO*-8 single"] = np.log10(df.loc[:, "bucket"].values)
            elif "multi" in f:
                delaylinedf["MSO*-8 multi"] = np.log10(df.loc[:, "delayline"].values)
                latticedf["MSO*-8 multi"] = np.log10(df.loc[:, "maglattice"].values)
                bucketdf["MSO*-8 multi"] = np.log10(df.loc[:, "bucket"].values)
        elif "four" in f:
            if "single" in f:
                delaylinedf["MSO*-4 single"] = np.log10(df.loc[:, "delayline"].values)
                latticedf["MSO*-4 single"] = np.log10(df.loc[:, "maglattice"].values)
                bucketdf["MSO*-4 single"] = np.log10(df.loc[:, "bucket"].values)
            elif "multi" in f:
                delaylinedf["MSO*-4 multi"] = np.log10(df.loc[:, "delayline"].values)
                latticedf["MSO*-4 multi"] = np.log10(df.loc[:, "maglattice"].values)
                bucketdf["MSO*-4 multi"] = np.log10(df.loc[:, "bucket"].values)
        elif "two" in f:
            if "single" in f:
                delaylinedf["MSO*-2 single"] = np.log10(df.loc[:, "delayline"].values)
                latticedf["MSO*-2 single"] = np.log10(df.loc[:, "maglattice"].values)
                bucketdf["MSO*-2 single"] = np.log10(df.loc[:, "bucket"].values)
            elif "multi" in f:
                delaylinedf["MSO*-2 multi"] = np.log10(df.loc[:, "delayline"].values)
                latticedf["MSO*-2 multi"] = np.log10(df.loc[:, "maglattice"].values)
                bucketdf["MSO*-2 multi"] = np.log10(df.loc[:, "bucket"].values)
        pass
    print("read files")
    
    sns.boxplot(data=delaylinedf, ax=delaylineax, notch=True, width=0.4, linewidth=1, fliersize=3, palette=materials_pal)
    delaylineax.set_xticklabels(delaylineax.get_xticklabels(),rotation=15)
    delaylineax.set_ylim(-4, 0)
    delaylineax.set_ylabel('log(NRMSE)')
    delaylineax.spines['right'].set_visible(False)
    delaylineax.spines['top'].set_visible(False)
    delaylinefig.tight_layout(pad=1)
    delaylinefig.suptitle("Ring Timescale Results")
    delaylinefig.savefig(path+"/MSO_ring.png", format="png")
    print("saved ring fig")

    sns.boxplot(data=latticedf, ax=latticeax, notch=True, width=0.4, linewidth=1, fliersize=3, palette=materials_pal)
    latticeax.set_xticklabels(latticeax.get_xticklabels(),rotation=15)
    latticeax.set_ylim(-4, 0)
    latticeax.set_ylabel('log(NRMSE)')
    latticeax.spines['right'].set_visible(False)
    latticeax.spines['top'].set_visible(False)
    latticefig.tight_layout(pad=1)
    latticefig.suptitle("Lattice Timescale Results")
    latticefig.savefig(path+"/MSO_lattice.png", format="png")
    print("saved lattice fig")
    sns.boxplot(data=bucketdf, ax=bucketax, notch=True, width=0.4, linewidth=1, fliersize=3, palette=materials_pal)
    bucketax.set_xticklabels(bucketax.get_xticklabels(),rotation=15)
    bucketax.set_ylim(-4, 0)
    bucketax.set_ylabel('log(NRMSE)')
    bucketax.spines['right'].set_visible(False)
    bucketax.spines['top'].set_visible(False)
    bucketfig.tight_layout(pad=1)
    bucketfig.suptitle("Bucket Timescale Results")
    bucketfig.savefig(path+"/MSO_bucket.png", format="png")
    print("saved bucket fig")
    return

# process_results()

# mso("/home/cw1647/phd/fakemat_experiments/paper_mso/results_chopped", 0, 0.6, "MSO")
# mso("/home/cw1647/phd/fakemat_experiments/paper_mso/results_chopped/single_timescale", 0, 1, "MSO single timescales")

# mso_alternate("/home/cw1647/phd/fakemat_experiments/paper_mso/results_chopped/multi_timescale", 0, 0.6, "MSO alternate")
# mso_alternate("/home/cw1647/phd/fakemat_experiments/paper_mso/results_chopped/single_timescale", 0, 1, "MSO single timescales alternate")

# mso_separately("/home/cw1647/phd/fakemat_experiments/paper_mso/results_chopped", "MSO separated")
# mso_separately("/home/cw1647/phd/fakemat_experiments/paper_mso/results_chopped/normalised_svd", "MSO separated")

mso_by_material("/home/cw1647/phd/fakemat_experiments/paper_mso/results/normalised_svd")
# mso_by_material("/home/cw1647/phd/fakemat_experiments/paper_mso/different_connections/results/normalised_svd")

# mso_by_material("/home/cw1647/phd/fakemat_experiments/paper_mso/ring_longer_training/normalised_svd")


