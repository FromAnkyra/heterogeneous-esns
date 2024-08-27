import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
# import het_reservoir_experiments.process_results as proc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})  


def plot_per_esn_type_single_material(path, name):
    pal = {'bucket': 'orange',
           'ring': 'orange',
           'lattice': 'orange'}
    sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})  
    single_material_large_results = pd.read_csv(path)
    heart_df = pd.DataFrame(columns=["bucket", "ring", "lattice"])
    respiration_df = pd.DataFrame(columns=["bucket", "ring", "lattice"])
    blood_df = pd.DataFrame(columns=["bucket", "ring", "lattice"])

    heart_df["bucket"] = single_material_large_results.loc[:, "bucket heart"]
    heart_df["ring"] = single_material_large_results.loc[:, "ring heart"]
    heart_df["lattice"] = single_material_large_results.loc[:, "lattice heart"]

    respiration_df["bucket"] = single_material_large_results.loc[:, "bucket chest"]
    respiration_df["ring"] = single_material_large_results.loc[:, "ring chest"]
    respiration_df["lattice"] = single_material_large_results.loc[:, "lattice chest"]

    blood_df["bucket"] = single_material_large_results.loc[:, "bucket blood"]
    blood_df["ring"] = single_material_large_results.loc[:, "ring blood"]
    blood_df["lattice"] = single_material_large_results.loc[:, "lattice blood"]

    fig, axes = plt.subplots(3, 1, figsize=(5, 9))
    sns.boxplot(data=heart_df, ax=axes[0], notch=True, width=0.3, linewidth=1, fliersize=3, palette=pal)
    sns.boxplot(data=respiration_df, ax=axes[1], notch=True, width=0.3, linewidth=1, fliersize=3, palette=pal)
    sns.boxplot(data=blood_df, ax=axes[2], notch=True, width=0.3, linewidth=1, fliersize=3, palette=pal)
    
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

    # fig.suptitle(name)
    fig.tight_layout(pad=1)
    newpath = f"{path.split('.')[0]}.png"
    print(newpath)
    fig.savefig(newpath)

    return

def plot_per_esn_multi_material(path, name):
    pal = {
        "B/R/L":"orange",
        "L/R/B":"orange"
    }
    results = pd.read_csv(path)

    heart_df = pd.DataFrame(columns=["B/R/L", "L/R/B"])
    respiration_df = pd.DataFrame(columns=["B/R/L", "L/R/B"])
    blood_df = pd.DataFrame(columns=["B/R/L", "L/R/B"])

    heart_df["B/R/L"] = results.loc[:, "brl heart"]
    heart_df["L/R/B"] = results.loc[:, "lrb heart"]

    respiration_df["B/R/L"] = results.loc[:, "brl chest"]
    respiration_df["L/R/B"] = results.loc[:, "lrb chest"]

    blood_df["B/R/L"] = results.loc[:, "brl blood"]
    blood_df["L/R/B"] = results.loc[:, "lrb blood"]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(5, 9))
    sns.boxplot(data=heart_df, ax=axes[0], notch=True, width=0.3, linewidth=1, fliersize=3, palette=pal)
    sns.boxplot(data=respiration_df, ax=axes[1], notch=True, width=0.3, linewidth=1, fliersize=3, palette=pal)
    sns.boxplot(data=blood_df, ax=axes[2], notch=True, width=0.3, linewidth=1, fliersize=3, palette=pal)
    
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

    # fig.suptitle(name)
    fig.tight_layout(pad=1)
    newpath = f"{path.split('.')[0]}.png"
    print(newpath)
    fig.savefig(newpath)
    return

def plot_full_comparison():
    # 3 rows (heart rate, respiration, blood oxygenation)
    #following groups of boxplots (all but the first three should have 2 boxplots, for all-to-all vs 1-to-1 mapping)
    # three colours (mapping not relevant (739AFF), 1-to-1 (DC267F), all-to-all(FFB000))
    ESN = [] # mapping not relevant
    rESN = [] # mapping not relevant
    single_reservoir = [] # mapping not relevant, lattice
    single_single = [] # all-to-all and 1-to-1, lattice
    single_material_multi_timescale = [] # all-to-all and 1-to-1, lattice
    multi_material_single_timescale = [] # all-to-all and 1-to-1, BRL
    multi_multi = [] # all-to-all and 1-to-1, BRL
    # total: 11 * 3 boxplots 

    
    return

def plot_oto_only(path):
    my_pal = {
        "esn": "#DC267F",
        "resn": "#DC267F",
        "single reservoir": "#DC267F",
        "single single one to one": "#FFB000", 
        "multi timescale one to one": "#FFB000", 
        "multi material one to one": "#FFB000",
        "multi multi one to one": "#FFB000",
        "blank1" : "orange"}

    filenames =[str(filepath) for filepath in Path(path).glob('*') if ".csv" in str(filepath) and "proportions" not in str(filepath)]
    heart_results = pd.DataFrame(columns = ["esn", "resn", "single reservoir", "blank1",
                                           "single single one to one", 
                                           "multi timescale one to one", 
                                           "multi material one to one", 
                                           "multi multi one to one"])
    respiration_results = pd.DataFrame(columns = ["esn", "resn", "single reservoir", "blank1",
                                           "single single one to one", 
                                           "multi timescale one to one", 
                                           "multi material one to one", 
                                           "multi multi one to one"])
    blood_results = pd.DataFrame(columns = ["esn", "resn", "single reservoir", "blank1",
                                           "single single one to one", 
                                           "multi timescale one to one", 
                                           "multi material one to one", 
                                           "multi multi one to one"])
    for item in filenames:
        df = pd.read_csv(item)
        name = item.split(".")[0]
        if "/esn_results" in name:
            heart_results["esn"] = df.loc[:, "heart"]
            respiration_results["esn"] = df.loc[:, "chest"]
            blood_results["esn"] = df.loc[:, "blood"]
        elif "resn_results" in name:
            heart_results["resn"] = df.loc[:, "heart"]
            respiration_results["resn"] = df.loc[:, "chest"]
            blood_results["resn"] = df.loc[:, "blood"]
        elif "single_reservoir" in name:
            heart_results["single reservoir"] = df.loc[:, "ring heart"]
            respiration_results["single reservoir"] = df.loc[:, "ring chest"]
            blood_results["single reservoir"] = df.loc[:, "ring blood"]
        elif "single_single_" in name:
            if "all_to_all" in name:
                pass
                # full_results["single single all to all"] = df.loc[:, "ring"]
            if "one_to_one" in name:
                heart_results["single single one to one"] = df.loc[:, "lattice heart"]
                respiration_results["single single one to one"] = df.loc[:, "lattice chest"]
                blood_results["single single one to one"] = df.loc[:, "lattice blood"]
                # full_results["single single one to one"] = df.loc[:, "ring"]
        elif "multi_timescale_" in name:
            if "all_to_all" in name:
                pass
                # full_results["multi timescale all to all"] = df.loc[:, "ring"]
            if "one_to_one" in name:
                # full_results["multi timescale one to one"] = df.loc[:, "ring"]
                heart_results["multi timescale one to one"] = df.loc[:, "lattice heart"]
                respiration_results["multi timescale one to one"] = df.loc[:, "lattice chest"]
                blood_results["multi timescale one to one"] = df.loc[:, "lattice blood"]
        elif "multi_material" in name:
            if "all_to_all" in name:
                pass
                # full_results["multi material all to all"] = df.loc[:, "rbl"]
            if "one_to_one" in name:
                # full_results["multi material one to one"] = df.loc[:, "brl"]
                heart_results["multi material one to one"] = df.loc[:, "brl heart"]
                respiration_results["multi material one to one"] = df.loc[:, "brl chest"]
                blood_results["multi material one to one"] = df.loc[:, "brl blood"]
        elif "multi_multi_" in name:
            if "all_to_all" in name:
                pass
                # full_results["multi multi all to all"] = df.loc[:, "rbl"]
            if "one_to_one" in name:
                # full_results["multi multi one to one"] = df.loc[:, "brl"]
                heart_results["multi multi one to one"] = df.loc[:, "lrb heart"]
                respiration_results["multi multi one to one"] = df.loc[:, "lrb chest"]
                blood_results["multi multi one to one"] = df.loc[:, "lrb blood"]
                pass
        # map_irrelevant = [esn, resn, single_reservoir]
        # all_to_all = [single_single_ata, multi_timescale_ata, multi_material_ata, multi_multi_ata]
        # one_to_one = [single_single_oto, multi_timescale_oto, multi_material_oto, multi_multi_oto]

        # colors = ["DC267F", "739AFF", "FFB000"]
        # groups = [map_irrelevant, all_to_all, one_to_one]
            
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(15, 6))
    sns.boxplot(data=heart_results, ax=axs[0], notch=True, width=0.3, linewidth=1, fliersize=3, palette=my_pal)
    sns.boxplot(data=respiration_results, ax=axs[1], notch=True, width=0.3, linewidth=1, fliersize=3, palette=my_pal)
    sns.boxplot(data=blood_results, ax=axs[2], notch=True, width=0.3, linewidth=1, fliersize=3, palette=my_pal)
    # axs.set_ylim(0, 0.8)
    labels_list = ["esn", "resn", "single reservoir", 
                   "single material,\nsingle timescale",
                   "single material,\nmulti timescale",
                   "multi material,\nsingle timescale",
                   "multi material,\nmulti timescale"]
    x_positions = [ 0, 1, 2, # positions for esn, resn & single reservoir
                    4, # sing. material, sing. timescale
                    5, # sing. material, mult. timescale
                    6, # mult. material, mult. timescale
                    7 # multi multi
                    ]
    axs[0].set_ylabel("NRMSE")
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].set_ylim(0, 1)
    axs[0].set_xlabel("heart rate results")
    axs[1].set_ylabel("NRMSE")
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel("respiration results")
    axs[2].set_ylabel("NRMSE")
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)
    axs[2].set_ylim(0, 1)
    axs[2].set_xlabel("blood oxygen results")

    axs[2].set_xticks(x_positions)
    axs[2].set_xticklabels(labels_list, rotation=25)

    width = 0.6
    plt.show()

    return

def plot_ata_only(path):
    
    my_pal = {
        "esn": "#DC267F",
        "resn": "#DC267F",
        "single reservoir": "#DC267F",
        "single single all to all": "#739AFF", 
        "multi timescale all to all": "#739AFF", 
        "multi material all to all": "#739AFF", 
        "multi multi all to all": "#739AFF",
        "blank1" : "orange"}

    filenames =[str(filepath) for filepath in Path(path).glob('*') if ".csv" in str(filepath) and "proportions" not in str(filepath)]
    heart_results = pd.DataFrame(columns = ["esn", "resn", "single reservoir", "blank1",
                                           "single single all to all", 
                                           "multi timescale all to all", 
                                           "multi material all to all", 
                                           "multi multi all to all"])
    respiration_results = pd.DataFrame(columns = ["esn", "resn", "single reservoir", "blank1",
                                           "single single all to all", 
                                           "multi timescale all to all", 
                                           "multi material all to all", 
                                           "multi multi all to all"])
    blood_results = pd.DataFrame(columns = ["esn", "resn", "single reservoir", "blank1",
                                           "single single all to all", 
                                           "multi timescale all to all", 
                                           "multi material all to all", 
                                           "multi multi all to all"])
    for item in filenames:
        df = pd.read_csv(item)
        name = item.split(".")[0]
        if "/esn_results" in name:
            heart_results["esn"] = df.loc[:, "heart"]
            respiration_results["esn"] = df.loc[:, "chest"]
            blood_results["esn"] = df.loc[:, "blood"]
        elif "resn_results" in name:
            heart_results["resn"] = df.loc[:, "heart"]
            respiration_results["resn"] = df.loc[:, "chest"]
            blood_results["resn"] = df.loc[:, "blood"]
        elif "single_reservoir" in name:
            heart_results["single reservoir"] = df.loc[:, "ring heart"]
            respiration_results["single reservoir"] = df.loc[:, "ring chest"]
            blood_results["single reservoir"] = df.loc[:, "ring blood"]
        elif "single_single_" in name:
            if "all_to_all" in name:
                heart_results["single single all to all"] = df.loc[:, "lattice heart"]
                respiration_results["single single all to all"] = df.loc[:, "lattice chest"]
                blood_results["single single all to all"] = df.loc[:, "lattice blood"]
            if "one_to_one" in name:
                pass
                # full_results["single single one to one"] = df.loc[:, "ring"]
        elif "multi_timescale_" in name:
            if "all_to_all" in name:
                heart_results["multi timescale all to all"] = df.loc[:, "lattice heart"]
                respiration_results["multi timescale all to all"] = df.loc[:, "lattice chest"]
                blood_results["multi timescale all to all"] = df.loc[:, "lattice blood"]
            if "one_to_one" in name:
                pass
                # full_results["multi timescale one to one"] = df.loc[:, "ring"]
        elif "multi_material" in name:
            if "all_to_all" in name:
                heart_results["multi material all to all"] = df.loc[:, "brl heart"]
                respiration_results["multi material all to all"] = df.loc[:, "brl chest"]
                blood_results["multi material all to all"] = df.loc[:, "brl blood"]
            if "one_to_one" in name:
                pass
                # full_results["multi material one to one"] = df.loc[:, "rbl"]
        elif "multi_multi_" in name:
            if "all_to_all" in name:
                heart_results["multi multi all to all"] = df.loc[:, "lrb heart"]
                respiration_results["multi multi all to all"] = df.loc[:, "lrb chest"]
                blood_results["multi multi all to all"] = df.loc[:, "lrb blood"]
            if "one_to_one" in name:
                # full_results["multi multi one to one"] = df.loc[:, "rbl"]
                pass

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(15, 6))
    sns.boxplot(data=heart_results, ax=axs[0], notch=True, width=0.3, linewidth=1, fliersize=3, palette=my_pal)
    sns.boxplot(data=respiration_results, ax=axs[1], notch=True, width=0.3, linewidth=1, fliersize=3, palette=my_pal)
    sns.boxplot(data=blood_results, ax=axs[2], notch=True, width=0.3, linewidth=1, fliersize=3, palette=my_pal)
    labels_list = ["esn", "resn", "single reservoir", 
                   "single material,\nsingle timescale",
                   "single material,\nmulti timescale",
                   "multi material,\nsingle timescale",
                   "multi material,\nmulti timescale"]
    x_positions = [ 0, 1, 2, # positions for esn, resn & single reservoir
                    4, # sing. material, sing. timescale
                    5, # sing. material, mult. timescale
                    6, # mult. material, mult. timescale
                    7 # multi multi
                    ]
    
    axs[0].set_ylabel("NRMSE")
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].set_ylim(0, 1)
    axs[0].set_xlabel("heart rate results")
    axs[1].set_ylabel("NRMSE")
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel("respiration results")
    axs[2].set_ylabel("NRMSE")
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)
    axs[2].set_ylim(0, 1)
    axs[2].set_xlabel("blood oxygen results")

    axs[2].set_xticks(x_positions)
    axs[2].set_xticklabels(labels_list, rotation=25)


    plt.show()
    return


#single reservoir
# plot_per_esn_type_single_material("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/single_reservoir.csv", "single reservoir")
#single material single timescale
# plot_per_esn_type_single_material("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/single_single_all_to_all.csv", "single material & timescale all-to-all")
# plot_per_esn_type_single_material("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/single_single_one_to_one.csv", "single material & timescale one-to-one")
#single material multi timescale
# plot_per_esn_type_single_material("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/multi_timescale_all_to_all.csv", "single mat, multi time all-to-all")
# plot_per_esn_type_single_material("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/multi_timescale_one_to_one.csv", "single mat, multi time one-to-one")

#multi material single timescale
# plot_per_esn_multi_material("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/multi_material_all_to_all.csv", "multi mat, single time all-to-all")
# plot_per_esn_multi_material("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/multi_material_one_to_one.csv", "multi mat, single time one-to-one")

# multi material multi timescale
# plot_per_esn_multi_material("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/multi_multi_all_to_all.csv", "multi mat, multi time all-to-all")
# plot_per_esn_multi_material("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results/multi_multi_one_to_one.csv", "multi mat, multi time one-to-one")

# plt.show()

plot_ata_only("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results_chopped")
plot_oto_only("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results_chopped")