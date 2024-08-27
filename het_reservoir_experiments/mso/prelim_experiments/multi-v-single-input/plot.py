import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# proc.process_results("/home/cw1647/phd/het_reservoir_experiments/mso/prelim_experiments/multi-v-single-input/results/")


my_pal = {
    "single input" : "#FFB000",
    "multi input" : "#739AFF"
}

sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})  

single_df = pd.read_csv("/home/cw1647/phd/het_reservoir_experiments/mso/prelim_experiments/multi-v-single-input/results/single-input.csv")
multi_df = pd.read_csv("/home/cw1647/phd/het_reservoir_experiments/mso/prelim_experiments/multi-v-single-input/results/multi-input-all-to-all.csv")

ring_df = pd.DataFrame(columns=["single input", "multi input"])
lattice_df = pd.DataFrame(columns=["single input", "multi input"])
bucket_df = pd.DataFrame(columns=["single input", "multi input"])

ring_df["single input"] = single_df.loc[:, "ring"]
ring_df["multi input"] = multi_df.loc[:, "ring"]

lattice_df["single input"] = single_df.loc[:, "lattice"]
lattice_df["multi input"] = multi_df.loc[:, "lattice"]

bucket_df["single input"] = single_df.loc[:, "bucket"]
bucket_df["multi input"] = multi_df.loc[:, "bucket"]

fig, ax = plt.subplots(1, 3)

sns.boxplot(data=ring_df, ax=ax[0], notch=True, width=0.4, linewidth=1, fliersize=3, palette=my_pal)
ax[0].set_ylim(0, 0.010)
ax[0].title.set_text("ring")
sns.boxplot(data=lattice_df, ax=ax[1], notch=True, width=0.4, linewidth=1, fliersize=3, palette=my_pal)
ax[1].set_ylim(0, 0.3)
ax[1].title.set_text("lattice")
sns.boxplot(data=bucket_df, ax=ax[2], notch=True, width=0.4, linewidth=1, fliersize=3, palette=my_pal)
ax[2].set_ylim(0, 1)
ax[2].title.set_text("bucket")
plt.show()