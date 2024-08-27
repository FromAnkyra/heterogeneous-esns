import seaborn as sns
import pandas as pd
import numpy as np
import het_reservoir_experiments.process_results as proc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

proc.process_results("/home/cw1647/phd/het_reservoir_experiments/mso/prelim_experiments/material-selection/results/")
pal= {
    "r/l/b": 'orange',
    "l/r/b": 'orange',
    "b/r/l" : 'orange',
    "b/l/r" : 'orange',
    "r/b/l": 'orange',
    "l/b/r": 'orange'
}

sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})  

results_df = pd.read_csv("/home/cw1647/phd/het_reservoir_experiments/mso/prelim_experiments/material-selection/results_chopped/material-selection.csv")

plot_df = pd.DataFrame(columns=["r/l/b", "l/r/b", "b/r/l", "b/l/r", "r/b/l", "l/b/r"])

plot_df["r/l/b"] = results_df.loc[:, "rlb"]
plot_df["l/r/b"] = results_df.loc[:, "lrb"]
plot_df["b/r/l"] = results_df.loc[:, "brl"]
plot_df["b/l/r"] = results_df.loc[:, "blr"]
plot_df["r/b/l"] = results_df.loc[:, "rbl"]
plot_df["l/b/r"] = results_df.loc[:, "lbr"]

fig, ax = plt.subplots(1, 1, figsize=(2*6, 4))

sns.boxplot(data=plot_df, ax=ax, notch=True, width=0.4, linewidth=1, fliersize=3, palette=pal)
ax.set_xticklabels(ax.get_xticklabels(),rotation=15)

ax.set_ylabel("NRMSE")

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout(pad=1)

plt.show()