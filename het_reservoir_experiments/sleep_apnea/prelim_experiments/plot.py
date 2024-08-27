import seaborn as sns
import pandas as pd
import numpy as np
import het_reservoir_experiments.process_results as proc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_results(path, name):
    pal= {
        "r/l/b": 'orange',
        "l/r/b": 'orange',
        "b/r/l" : 'orange',
        "b/l/r" : 'orange',
        "r/b/l": 'orange',
        "l/b/r": 'orange'
    }

    sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})  

    results_rlb = pd.read_csv(f"{path}/rlb.csv")
    results_lrb = pd.read_csv(f"{path}/lrb.csv")
    results_brl = pd.read_csv(f"{path}/brl.csv")
    results_blr = pd.read_csv(f"{path}/blr.csv")
    results_rbl = pd.read_csv(f"{path}/rbl.csv")
    results_lbr = pd.read_csv(f"{path}/lbr.csv")


    heart_df = pd.DataFrame(columns=["r/l/b", "l/r/b", "b/r/l", "b/l/r", "r/b/l", "l/b/r"])
    chest_df = pd.DataFrame(columns=["r/l/b", "l/r/b", "b/r/l", "b/l/r", "r/b/l", "l/b/r"])
    blood_df = pd.DataFrame(columns=["r/l/b", "l/r/b", "b/r/l", "b/l/r", "r/b/l", "l/b/r"])

    fig, axes = plt.subplots(3, 1, figsize=(2*6, 4))

    heart_df["r/l/b"] = results_rlb.loc[:, "heart"]
    heart_df["l/r/b"] = results_lrb.loc[:, "heart"]
    heart_df["b/r/l"] = results_brl.loc[:, "heart"]
    heart_df["b/l/r"] = results_blr.loc[:, "heart"]
    heart_df["r/b/l"] = results_rbl.loc[:, "heart"]
    heart_df["l/b/r"] = results_lbr.loc[:, "heart"]

    chest_df["r/l/b"] = results_rlb.loc[:, "chest"]
    chest_df["l/r/b"] = results_lrb.loc[:, "chest"]
    chest_df["b/r/l"] = results_brl.loc[:, "chest"]
    chest_df["b/l/r"] = results_blr.loc[:, "chest"]
    chest_df["r/b/l"] = results_rbl.loc[:, "chest"]
    chest_df["l/b/r"] = results_lbr.loc[:, "chest"]

    blood_df["r/l/b"] = results_rlb.loc[:, "blood"]
    blood_df["l/r/b"] = results_lrb.loc[:, "blood"]
    blood_df["b/r/l"] = results_brl.loc[:, "blood"]
    blood_df["b/l/r"] = results_blr.loc[:, "blood"]
    blood_df["r/b/l"] = results_rbl.loc[:, "blood"]
    blood_df["l/b/r"] = results_lbr.loc[:, "blood"]

    sns.boxplot(data=heart_df, ax=axes[0], notch=True, width=0.4, linewidth=1, fliersize=3, palette=pal)
    sns.boxplot(data=chest_df, ax=axes[1], notch=True, width=0.4, linewidth=1, fliersize=3, palette=pal)
    sns.boxplot(data=blood_df, ax=axes[2], notch=True, width=0.4, linewidth=1, fliersize=3, palette=pal)

    axes[0].set_ylabel("NRMSE")
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("heart rate results")

    axes[1].set_ylabel("NRMSE")
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("respiration results")

    axes[2].set_ylabel("NRMSE")
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("blood oxygen results")

    fig.suptitle(name)
    fig.tight_layout(pad=1)

    return 

proc.process_results("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/fast-to-slow")
proc.process_results("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/slow-to-slow/")

plot_results("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results_chopped/fast-to-slow", "fast to slow")
plot_results("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results_chopped/slow-to-slow", "slow to slow")

plt.show()