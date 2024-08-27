import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})  

def plot_per_esn_type_single_material(path, name):
    pal = {'bucket': 'orange',
           'ring': 'orange',
           'lattice': 'orange'}
    sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})  
    single_material_large_results = pd.read_csv(path)
    results_df = pd.DataFrame(columns=["bucket", "ring", "lattice"])

    results_df["bucket"] = single_material_large_results.loc[:, "bucket"]
    results_df["ring"] = single_material_large_results.loc[:, "ring"]
    results_df["lattice"] = single_material_large_results.loc[:, "lattice"]

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.boxplot(data=results_df, ax=ax, notch=True, width=0.3, linewidth=1, fliersize=3, palette=pal)
    
    ax.set_ylabel("NRMSE")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(0, 1)

    # fig.suptitle(name)
    fig.tight_layout(pad=1)
    newpath = f"{path.split('.')[0]}.png"
    print(newpath)
    fig.savefig(newpath)

    return

def plot_per_esn_multi_material(path, name):
    pal = {
        "R/L/B":"orange",
        "R/B/L":"orange"
    }
    results = pd.read_csv(path)

    results_df = pd.DataFrame(columns=["R/L/B", "R/B/L"])

    results_df["R/L/B"] = results.loc[:, "rlb"]
    results_df["R/B/L"] = results.loc[:, "rbl"]

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.boxplot(data=results_df, ax=ax, notch=True, width=0.6, linewidth=1, fliersize=3, palette=pal)

    ax.set_ylabel("NRMSE")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(0, 1)
    

    # fig.suptitle(name)
    fig.tight_layout(pad=1)
    newpath = f"{path.split('.')[0]}.png"
    print(newpath)
    fig.savefig(newpath)
    return

def plot_all(path):
    # ESN
    # rESN
    # Single reservoir ring (DC267F)
    # single mat, single timescale ring (ata (739AFF)/oto (FFB000))
    # single mat, multi time ring (ata/oto)
    # multi mat, single time R/B/L (ata/oto)
    # multi mat, multi time R/B/L (ata/oto)

    my_pal = {
        "esn": "#DC267F",
        "resn": "#DC267F",
        "single reservoir": "#DC267F",
        "single single all to all": "#739AFF", 
        "single single one to one": "#FFB000", 
        "multi timescale all to all": "#739AFF", 
        "multi timescale one to one": "#FFB000", 
        "multi material all to all": "#739AFF", 
        "multi material one to one": "#FFB000",
        "multi multi all to all": "#739AFF",
        "multi multi one to one": "#FFB000",
        "blank1" : "orange",
        "blank2" : "orange",
        "blank3" : "orange", 
        "blank4" : "orange" }

    filenames =[str(filepath) for filepath in Path(path).glob('*') if ".csv" in str(filepath)]
    full_results = pd.DataFrame(columns = ["esn", "resn", "single reservoir", "blank1",
                                           "single single all to all", 
                                           "single single one to one", "blank2",
                                           "multi timescale all to all", 
                                           "multi timescale one to one", "blank3",
                                           "multi material all to all", 
                                           "multi material one to one", "blank4",
                                           "multi multi all to all",
                                           "multi multi one to one"])
    for item in filenames:
        df = pd.read_csv(item)
        name = item.split(".")[0]
        if "/esn_results" in name:
            full_results["esn"] = df.loc[:, "test"]
        elif "resn_results" in name:
            full_results["resn"] = df.loc[:, "test"]
        elif "single_reservoir" in name:
            full_results["single reservoir"] = df.loc[:, "ring"]
        elif "single_single_" in name:
            if "all_to_all" in name:
                full_results["single single all to all"] = df.loc[:, "ring"]
            if "one_to_one" in name:
                full_results["single single one to one"] = df.loc[:, "ring"]
        elif "multi_timescale_" in name:
            if "all_to_all" in name:
                full_results["multi timescale all to all"] = df.loc[:, "ring"]
            if "one_to_one" in name:
                full_results["multi timescale one to one"] = df.loc[:, "ring"]
        elif "multi_material" in name:
            if "all_to_all" in name:
                full_results["multi material all to all"] = df.loc[:, "rbl"]
            if "one_to_one" in name:
                full_results["multi material one to one"] = df.loc[:, "rbl"]
        elif "multi_multi_" in name:
            if "all_to_all" in name:
                full_results["multi multi all to all"] = df.loc[:, "rbl"]
            if "one_to_one" in name:
                full_results["multi multi one to one"] = df.loc[:, "rbl"]
                pass
        # map_irrelevant = [esn, resn, single_reservoir]
        # all_to_all = [single_single_ata, multi_timescale_ata, multi_material_ata, multi_multi_ata]
        # one_to_one = [single_single_oto, multi_timescale_oto, multi_material_oto, multi_multi_oto]

        # colors = ["DC267F", "739AFF", "FFB000"]
        # groups = [map_irrelevant, all_to_all, one_to_one]
            
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    sns.boxplot(data=full_results, ax=ax, notch=True, width=0.3, linewidth=1, fliersize=3, palette=my_pal)
    ax.set_ylim(0, 0.8)
    labels_list = ["esn", "resn", "single reservoir", 
                   "single material,\nsingle timescale",
                   "single material,\nmulti timescale",
                   "multi material,\nsingle timescale",
                   "multi material,\nmulti timescale"]
    x_positions = [ 0, 1, 2, # positions for esn, resn & single reservoir
                    4.5, # sing. material, sing. timescale
                    7.5, # sing. material, mult. timescale
                    10.5, # mult. material, mult. timescale
                    13.5 # multi multi
                    ]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels_list)
    plt.show()
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

    filenames =[str(filepath) for filepath in Path(path).glob('*') if ".csv" in str(filepath)]
    full_results = pd.DataFrame(columns = ["esn", "resn", "single reservoir", "blank1",
                                           "single single one to one", 
                                           "multi timescale one to one", 
                                           "multi material one to one", 
                                           "multi multi one to one"])
    for item in filenames:
        df = pd.read_csv(item)
        name = item.split(".")[0]
        if "/esn_results" in name:
            full_results["esn"] = df.loc[:, "test"]
        elif "resn_results" in name:
            full_results["resn"] = df.loc[:, "test"]
        elif "single_reservoir" in name:
            full_results["single reservoir"] = df.loc[:, "ring"]
        elif "single_single_" in name:
            if "all_to_all" in name:
                pass
                # full_results["single single all to all"] = df.loc[:, "ring"]
            if "one_to_one" in name:
                full_results["single single one to one"] = df.loc[:, "ring"]
        elif "multi_timescale_" in name:
            if "all_to_all" in name:
                pass
                # full_results["multi timescale all to all"] = df.loc[:, "ring"]
            if "one_to_one" in name:
                full_results["multi timescale one to one"] = df.loc[:, "bucket"]
        elif "multi_material" in name:
            if "all_to_all" in name:
                pass
                # full_results["multi material all to all"] = df.loc[:, "rbl"]
            if "one_to_one" in name:
                full_results["multi material one to one"] = df.loc[:, "rbl"]
        elif "multi_multi_" in name:
            if "all_to_all" in name:
                pass
                # full_results["multi multi all to all"] = df.loc[:, "rbl"]
            if "one_to_one" in name:
                full_results["multi multi one to one"] = df.loc[:, "rbl"]
                pass
        # map_irrelevant = [esn, resn, single_reservoir]
        # all_to_all = [single_single_ata, multi_timescale_ata, multi_material_ata, multi_multi_ata]
        # one_to_one = [single_single_oto, multi_timescale_oto, multi_material_oto, multi_multi_oto]

        # colors = ["DC267F", "739AFF", "FFB000"]
        # groups = [map_irrelevant, all_to_all, one_to_one]
            
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    sns.boxplot(data=full_results, ax=ax, notch=True, width=0.3, linewidth=1, fliersize=3, palette=my_pal)
    ax.set_ylim(0, 0.8)
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
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels_list, rotation=25)
    
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

    filenames =[str(filepath) for filepath in Path(path).glob('*') if ".csv" in str(filepath)]
    full_results = pd.DataFrame(columns = ["esn", "resn", "single reservoir", "blank1",
                                           "single single all to all", 
                                           "multi timescale all to all", 
                                           "multi material all to all", 
                                           "multi multi all to all"])
    for item in filenames:
        df = pd.read_csv(item)
        name = item.split(".")[0]
        if "/esn_results" in name:
            full_results["esn"] = df.loc[:, "test"]
        elif "resn_results" in name:
            full_results["resn"] = df.loc[:, "test"]
        elif "single_reservoir" in name:
            full_results["single reservoir"] = df.loc[:, "ring"]
        elif "single_single_" in name:
            if "all_to_all" in name:
                full_results["single single all to all"] = df.loc[:, "ring"]
            if "one_to_one" in name:
                pass
                # full_results["single single one to one"] = df.loc[:, "ring"]
        elif "multi_timescale_" in name:
            if "all_to_all" in name:
                full_results["multi timescale all to all"] = df.loc[:, "ring"]
            if "one_to_one" in name:
                pass
                # full_results["multi timescale one to one"] = df.loc[:, "ring"]
        elif "multi_material" in name:
            if "all_to_all" in name:
                full_results["multi material all to all"] = df.loc[:, "rbl"]
            if "one_to_one" in name:
                pass
                # full_results["multi material one to one"] = df.loc[:, "rbl"]
        elif "multi_multi_" in name:
            if "all_to_all" in name:
                full_results["multi multi all to all"] = df.loc[:, "rbl"]
            if "one_to_one" in name:
                # full_results["multi multi one to one"] = df.loc[:, "rbl"]
                pass
            
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    sns.boxplot(data=full_results, ax=ax, notch=True, width=0.3, linewidth=1, fliersize=3, palette=my_pal)
    ax.set_ylim(0, 1)
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
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels_list, rotation=25)
    plt.show()
    return

def plot_oto_no_basecase(path):
    my_pal = {
        "esn": "#DC267F",
        "resn": "#DC267F",
        "single reservoir": "#DC267F",
        "single single one to one": "#FFB000", 
        "multi timescale one to one": "#FFB000", 
        "multi material one to one": "#FFB000",
        "multi multi one to one": "#FFB000",
        "blank1" : "orange"}

    filenames =[str(filepath) for filepath in Path(path).glob('*') if ".csv" in str(filepath)]
    full_results = pd.DataFrame(columns = ["single single one to one", 
                                           "multi timescale one to one", 
                                           "multi material one to one", 
                                           "multi multi one to one"])
    for item in filenames:
        df = pd.read_csv(item)
        name = item.split(".")[0]
        # if "/esn_results" in name:
        #     full_results["esn"] = df.loc[:, "test"]
        # elif "resn_results" in name:
        #     full_results["resn"] = df.loc[:, "test"]
        # if "single_reservoir" in name:
        #     full_results["single reservoir"] = df.loc[:, "ring"]
        if "single_single_" in name:
            if "all_to_all" in name:
                pass
                # full_results["single single all to all"] = df.loc[:, "ring"]
            if "one_to_one" in name:
                full_results["single single one to one"] = df.loc[:, "ring"]
        elif "multi_timescale_" in name:
            if "all_to_all" in name:
                pass
                # full_results["multi timescale all to all"] = df.loc[:, "ring"]
            if "one_to_one" in name:
                full_results["multi timescale one to one"] = df.loc[:, "ring"]
        elif "multi_material" in name:
            if "all_to_all" in name:
                pass
                # full_results["multi material all to all"] = df.loc[:, "rbl"]
            if "one_to_one" in name:
                full_results["multi material one to one"] = df.loc[:, "rbl"]
        elif "multi_multi_" in name:
            if "all_to_all" in name:
                pass
                # full_results["multi multi all to all"] = df.loc[:, "rbl"]
            if "one_to_one" in name:
                full_results["multi multi one to one"] = df.loc[:, "rbl"]
                pass
        # map_irrelevant = [esn, resn, single_reservoir]
        # all_to_all = [single_single_ata, multi_timescale_ata, multi_material_ata, multi_multi_ata]
        # one_to_one = [single_single_oto, multi_timescale_oto, multi_material_oto, multi_multi_oto]

        # colors = ["DC267F", "739AFF", "FFB000"]
        # groups = [map_irrelevant, all_to_all, one_to_one]
            
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    sns.boxplot(data=full_results, ax=ax, notch=True, width=0.3, linewidth=1, fliersize=3, palette=my_pal)
    ax.set_ylim(0, 0.8)
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
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels_list)
    
    width = 0.6
    plt.show()

    return

def plot_ata_no_basecase(path):
    
    my_pal = {
        "esn": "#DC267F",
        "resn": "#DC267F",
        "single reservoir": "#DC267F",
        "single single all to all": "#739AFF", 
        "multi timescale all to all": "#739AFF", 
        "multi material all to all": "#739AFF", 
        "multi multi all to all": "#739AFF",
        "blank1" : "orange"}

    filenames =[str(filepath) for filepath in Path(path).glob('*') if ".csv" in str(filepath)]
    full_results = pd.DataFrame(columns = ["single single all to all", 
                                           "multi timescale all to all", 
                                           "multi material all to all", 
                                           "multi multi all to all"])
    for item in filenames:
        df = pd.read_csv(item)
        name = item.split(".")[0]
        # if "/esn_results" in name:
        #     full_results["esn"] = df.loc[:, "test"]
        # elif "resn_results" in name:
        #     full_results["resn"] = df.loc[:, "test"]
        # elif "single_reservoir" in name:
        #     full_results["single reservoir"] = df.loc[:, "ring"]
        if "single_single_" in name:
            if "all_to_all" in name:
                full_results["single single all to all"] = df.loc[:, "ring"]
            if "one_to_one" in name:
                pass
                # full_results["single single one to one"] = df.loc[:, "ring"]
        elif "multi_timescale_" in name:
            if "all_to_all" in name:
                full_results["multi timescale all to all"] = df.loc[:, "ring"]
            if "one_to_one" in name:
                pass
                # full_results["multi timescale one to one"] = df.loc[:, "ring"]
        elif "multi_material" in name:
            if "all_to_all" in name:
                full_results["multi material all to all"] = df.loc[:, "rbl"]
            if "one_to_one" in name:
                pass
                # full_results["multi material one to one"] = df.loc[:, "rbl"]
        elif "multi_multi_" in name:
            if "all_to_all" in name:
                full_results["multi multi all to all"] = df.loc[:, "rbl"]
            if "one_to_one" in name:
                # full_results["multi multi one to one"] = df.loc[:, "rbl"]
                pass
            
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.boxplot(data=full_results, ax=ax, notch=True, width=0.3, linewidth=1, fliersize=3, palette=my_pal)
    ax.set_ylim(0, 0.15)
    labels_list = ["single material,\nsingle timescale",
                   "single material,\nmulti timescale",
                   "multi material,\nsingle timescale",
                   "multi material,\nmulti timescale"]
    x_positions = [ 0, # sing. material, sing. timescale
                    1, # sing. material, mult. timescale
                    2, # mult. material, mult. timescale
                    3 # multi multi
                    ]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels_list, rotation=25)
    plt.show()
    return


#single reservoir
# plot_per_esn_type_single_material("/home/cw1647/phd/het_reservoir_experiments/mso/results_chopped/single_reservoir.csv", "single reservoir")
# #single material single timescale
# plot_per_esn_type_single_material("/home/cw1647/phd/het_reservoir_experiments/mso/results_chopped/single_single_all_to_all.csv", "single material & timescale all-to-all")
# plot_per_esn_type_single_material("/home/cw1647/phd/het_reservoir_experiments/mso/results_chopped/single_single_one_to_one.csv", "single material & timescale one-to-one")
# #single material multi timescale
# plot_per_esn_type_single_material("/home/cw1647/phd/het_reservoir_experiments/mso/results_chopped/multi_timescale_all_to_all.csv", "single mat, multi time all-to-all")
# plot_per_esn_type_single_material("/home/cw1647/phd/het_reservoir_experiments/mso/results_chopped/multi_timescale_one_to_one.csv", "single mat, multi time one-to-one")

# #multi material single timescale
# plot_per_esn_multi_material("/home/cw1647/phd/het_reservoir_experiments/mso/results_chopped/multi_material_all_to_all.csv", "multi mat, single time all-to-all")
# plot_per_esn_multi_material("/home/cw1647/phd/het_reservoir_experiments/mso/results_chopped/multi_material_one_to_one.csv", "multi mat, single time one-to-one")

# # multi material multi timescale
# plot_per_esn_multi_material("/home/cw1647/phd/het_reservoir_experiments/mso/results_chopped/multi_multi_all_to_all.csv", "multi mat, multi time all-to-all")
# plot_per_esn_multi_material("/home/cw1647/phd/het_reservoir_experiments/mso/results_chopped/multi_multi_one_to_one.csv", "multi mat, multi time one-to-one")

# plot_all("/home/cw1647/phd/het_reservoir_experiments/mso/results")
# plot_ata_only("/home/cw1647/phd/het_reservoir_experiments/mso/results")
# plot_ata_no_basecase("/home/cw1647/phd/het_reservoir_experiments/mso/results")
plot_oto_only("/home/cw1647/phd/het_reservoir_experiments/mso/results")