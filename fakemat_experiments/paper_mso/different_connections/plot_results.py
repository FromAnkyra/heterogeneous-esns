import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sns.set_context(rc={'font.size': 15, 'axes.titlesize': 15, 'axes.labelsize': 15})
def mso_by_material(path):
    materials_pal = {
        'single timescale': 'blue',
        'total sleep mode': 'orange',
        'input sleep mode': 'orange',
        'output sleep mode*': 'orange'
    }
    filenames = [str(path) for path in Path(path).glob('*') if "csv" in str(path)]
    delaylinefig, delaylineax = plt.subplots(1, 1, figsize=(4* 2, 4))

    sns.set_context(rc={'font.size': 15, 'axes.titlesize': 15, 'axes.labelsize': 15})  
    delaylinedf = pd.DataFrame(columns=["single timescale", 'total sleep mode', 'input sleep mode', 'output sleep mode*']) 

    for f in filenames:
        print(f)
        df = pd.read_csv(f)
        if "bucket" in f:
            if "single" in f:
                delaylinedf["single timescale"] = np.log10(df.loc[:, "bucket"].values)
            elif "multi" in f:
                delaylinedf['output sleep mode*'] = np.log10(df.loc[:, "bucket"].values)
        elif "ring" in f:
            if "single" in f:
                delaylinedf["single timescale"] = np.log10(df.loc[:, "bucket"].values)
            elif "multi" in f:
                delaylinedf['total sleep mode'] = np.log10(df.loc[:, "bucket"].values)
        elif "lattice" in f:
            if "single" in f:
                print(f"{f=}")
                delaylinedf["single timescale"] = np.log10(df.loc[:, "bucket"].values)
            elif "multi" in f:
                print(f"{f=}")
                delaylinedf['input sleep mode'] = np.log10(df.loc[:, "bucket"].values)
        pass
    
    
    sns.boxplot(data=delaylinedf, ax=delaylineax, notch=True, width=0.4, linewidth=1, fliersize=3, palette=materials_pal)
    delaylineax.set_xticklabels(delaylineax.get_xticklabels(),rotation=15)
    delaylineax.set_ylim(-4, 0)
    delaylineax.set_ylabel('log(NRMSE)')
    delaylineax.spines['right'].set_visible(False)
    delaylineax.spines['top'].set_visible(False)

    delaylinefig.tight_layout(pad=1)
    delaylinefig.suptitle("Bucket Sleep Mode Results")
    delaylinefig.savefig(path+"/diff_sleep_modes_bucket.png", format="png")

    return


mso_by_material("/home/cw1647/phd/fakemat_experiments/paper_mso/different_connections/results/normalised_svd")


