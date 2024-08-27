import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def process_results():
    filenames =[str(path) for path in Path("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results/").glob('*') if ".csv" in str(path)]
    print(filenames)
    i = 0
    def f(item):
        nonlocal i
        if item > 1:
            i+=1
            return 1.0
        else: 
            return item
    vf = np.vectorize(f)


    for file in filenames:
        print(file)
        df = pd.read_csv(file)
        newdf = pd.DataFrame(columns=list(df.columns))
        chopped_prop = pd.DataFrame(columns=list(df.columns))
        for col in df.columns:
            i=0
            data = df.loc[:, col].values
            newdf[col] = vf(data)
            chopped_prop.at[0, col] = i
            pass
        pieces = file.split("results")
        new_filename = pieces[0]+"results_chopped"+pieces[1]
        pieces_plus = pieces[1].split(".")
        proportions = f"{pieces[0]}results_chopped{pieces_plus[0]}_proportions.{pieces_plus[1]}"
        print(new_filename)
        newdf.to_csv(new_filename)  
        chopped_prop.to_csv(proportions)
    return

my_pal = {'bucket': 'blue',
          'bblood' : 'blue',
          'bchest' : 'blue',
          'bheart': 'blue',
          'ring': 'green',
          'dblood' : 'green',
          'dchest' : 'green',
          'dheart' : 'green',
          'lattice': 'gold',
          'mlblood':'gold',
          'mlchest':'gold',
          'mlheart':'gold',
          'mixed':'darkorange',
          'mblood': 'darkorange',
          'mchest': 'darkorange',
          'mheart': 'darkorange',}
sns.set_context(rc={'font.size': 18, 'axes.titlesize': 28, 'axes.labelsize': 28})


def sleep_apnea(path, low, high, title):
    print(path)
    errordf = pd.read_csv(path)
    fig, ax = plt.subplots(1, 1, figsize=(4 * 4, 5))    
    sns.boxplot(data=errordf, ax=ax, notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    ax.set_ylabel('NRMSE')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(low, high)
    fig.suptitle(title)
    fig.savefig(path.split(".")[0]+".png", format="png")
    return

def sleep_apnea_alternate(path, low, high, title):
    print(path)
    errordf = pd.DataFrame(columns=["bheart", "dheart", "mlheart", "mheart", "bchest", "dchest", "mlchest", "mchest", "bblood", "dblood", "mlblood", "mblood"])
    olddf = pd.read_csv(path)
    errordf["bheart"] = olddf.loc[:, "bheart"].values
    errordf["dheart"] = olddf.loc[:, "dheart"].values
    errordf["mlheart"] = olddf.loc[:, "mlheart"].values
    errordf["mheart"] = olddf.loc[:, "mheart"].values
    errordf["bchest"] = olddf.loc[:, "bchest"].values
    errordf["dchest"] = olddf.loc[:, "dchest"].values
    errordf["mlchest"] = olddf.loc[:, "mlchest"].values
    errordf["mchest"] = olddf.loc[:, "mchest"].values
    errordf["bblood"] = olddf.loc[:, "bblood"].values
    errordf["dblood"] = olddf.loc[:, "dblood"].values
    errordf["mlblood"] = olddf.loc[:, "mlblood"].values
    errordf["mblood"] = olddf.loc[:, "mblood"].values

    fig, ax = plt.subplots(1, 1, figsize=(4 * 4, 5))    
    sns.set_context(rc={'font.size': 18, 'axes.titlesize': 28, 'axes.labelsize': 28})
    sns.boxplot(data=errordf, ax=ax, notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    ax.set_ylabel('NRMSE')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(low, high)
    fig.suptitle(title)
    fig.savefig(path.split(".")[0]+"_alternate.png", format="png")

def sleep_apnea_separately(path, title):
    errordf = pd.read_csv(path)
    fig, ax = plt.subplots(1, 3, figsize=(6 * 4, 6))  
    sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})    
    heartdf = pd.DataFrame(columns=["bucket", "ring", "lattice"])
    chestdf = pd.DataFrame(columns=["bucket", "ring", "lattice"])
    blooddf = pd.DataFrame(columns=["bucket", "ring", "lattice"])

    heartdf["bucket"] = errordf.loc[:, "bheart"]
    heartdf["ring"] = errordf.loc[:, "dheart"]
    heartdf["lattice"] = errordf.loc[:, "mlheart"]
    # heartdf["mixed"] = errordf.loc[:, "mheart"]
    low = max(min(heartdf.loc[:, "bucket"].min()-0.1, heartdf.loc[:, "ring"].min()-0.1, heartdf.loc[:, "lattice"].min()-0.1), 0)
    high = min(max(heartdf.loc[:, "bucket"].max()+0.1, heartdf.loc[:, "ring"].max()+0.1, heartdf.loc[:, "lattice"].max()+0.1), 1)
    sns.boxplot(data=heartdf, ax=ax[0], notch=True, width=0.2, linewidth=1, fliersize=3, palette=my_pal)
    ax[0].set_ylabel('NRMSE')
    ax[0].set_xlabel('heart rate')
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].set_ylim(0, high)

    chestdf["bucket"] = errordf.loc[:, "bchest"]
    chestdf["ring"] = errordf.loc[:, "dchest"]
    chestdf["lattice"] = errordf.loc[:, "mlchest"]
    low = max(min(chestdf.loc[:, "bucket"].min()-0.1, chestdf.loc[:, "ring"].min()-0.1, chestdf.loc[:, "lattice"].min()-0.1), 0)
    high = min(max(chestdf.loc[:, "bucket"].max()+0.1, chestdf.loc[:, "ring"].max()+0.1, chestdf.loc[:, "lattice"].max()+0.1), 1)
    sns.boxplot(data=chestdf, ax=ax[1], notch=True, width=0.2, linewidth=1, fliersize=3, palette=my_pal)
    ax[1].set_ylabel('NRMSE')
    ax[1].set_xlabel('respiration')
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_ylim(0, high)

    blooddf["bucket"] = errordf.loc[:, "bblood"]
    blooddf["ring"] = errordf.loc[:, "dblood"]
    blooddf["lattice"] = errordf.loc[:, "mlblood"]
    low = max(min(blooddf.loc[:, "bucket"].min()-0.1, blooddf.loc[:, "ring"].min()-0.1, blooddf.loc[:, "lattice"].min()-0.1), 0)
    high = min(max(blooddf.loc[:, "bucket"].max()+0.1, blooddf.loc[:, "ring"].max()+0.1, blooddf.loc[:, "lattice"].max()+0.1), 1)
    sns.boxplot(data=blooddf, ax=ax[2], notch=True, width=0.2, linewidth=1, fliersize=3, palette=my_pal)
    ax[2].set_ylabel('NRMSE')
    ax[2].set_xlabel('blood oxygenation')
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['top'].set_visible(False)
    ax[2].set_ylim(0, high)
    fig.tight_layout(pad=1)
    # fig.suptitle(title)
    fig.savefig(path.split(".")[0]+"_by_input.png", format="png")
    return

def plot_training_lengths():
    heart_ring_df = pd.DataFrame(columns=["500", "750", "1000", "1250", "1500", "1750", "2000"])
    heart_lattice_df = pd.DataFrame(columns=["500", "750", "1000", "1250", "1500", "1750", "2000"])
    heart_bucket_df = pd.DataFrame(columns=["500", "750", "1000", "1250", "1500", "1750", "2000"])
    heart_mixed_df = pd.DataFrame(columns=["500", "750", "1000", "1250", "1500", "1750", "2000"])
    
    chest_ring_df = pd.DataFrame(columns=["500", "750", "1000", "1250", "1500", "1750", "2000"])
    chest_lattice_df = pd.DataFrame(columns=["500", "750", "1000", "1250", "1500", "1750", "2000"])
    chest_bucket_df = pd.DataFrame(columns=["500", "750", "1000", "1250", "1500", "1750", "2000"])
    chest_mixed_df = pd.DataFrame(columns=["500", "750", "1000", "1250", "1500", "1750", "2000"])

    blood_ring_df = pd.DataFrame(columns=["500", "750", "1000", "1250", "1500", "1750", "2000"])
    blood_lattice_df = pd.DataFrame(columns=["500", "750", "1000", "1250", "1500", "1750", "2000"])
    blood_bucket_df = pd.DataFrame(columns=["500", "750", "1000", "1250", "1500", "1750", "2000"])
    blood_mixed_df = pd.DataFrame(columns=["500", "750", "1000", "1250", "1500", "1750", "2000"])

    heartfig, heartax = plt.subplots(4, 1)
    chestfig, chestax = plt.subplots(4, 1)
    bloodfig, bloodax = plt.subplots(4, 1)

    filenames =[str(path) for path in Path("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped").glob('*') if ".csv" in str(path) and "proportions" not in str(path)]
    for f in filenames:
        if "_500" in f:
            olddf = pd.read_csv(f)
            heart_ring_df["500"] = olddf.loc[:, "dheart"]
            chest_ring_df["500"] = olddf.loc[:, "dchest"]
            blood_ring_df["500"] = olddf.loc[:, "dblood"]

            heart_lattice_df["500"] = olddf.loc[:, "mlheart"]
            chest_lattice_df["500"] = olddf.loc[:, "mlchest"]
            blood_lattice_df["500"] = olddf.loc[:, "mlblood"]

            heart_bucket_df["500"] = olddf.loc[:, "bheart"]
            chest_bucket_df["500"] = olddf.loc[:, "bchest"]
            blood_bucket_df["500"] = olddf.loc[:, "bblood"]

            heart_mixed_df["500"] = olddf.loc[:, "mheart"]
            chest_mixed_df["500"] = olddf.loc[:, "mchest"]
            blood_mixed_df["500"] = olddf.loc[:, "mblood"]

        elif "_750" in f:
            olddf = pd.read_csv(f)
            heart_ring_df["750"] = olddf.loc[:, "dheart"]
            chest_ring_df["750"] = olddf.loc[:, "dchest"]
            blood_ring_df["750"] = olddf.loc[:, "dblood"]

            heart_lattice_df["750"] = olddf.loc[:, "mlheart"]
            chest_lattice_df["750"] = olddf.loc[:, "mlchest"]
            blood_lattice_df["750"] = olddf.loc[:, "mlblood"]

            heart_bucket_df["750"] = olddf.loc[:, "bheart"]
            chest_bucket_df["750"] = olddf.loc[:, "bchest"]
            blood_bucket_df["750"] = olddf.loc[:, "bblood"]

            heart_mixed_df["750"] = olddf.loc[:, "mheart"]
            chest_mixed_df["750"] = olddf.loc[:, "mchest"]
            blood_mixed_df["750"] = olddf.loc[:, "mblood"]
        elif "_1000" in f:
            olddf = pd.read_csv(f)
            heart_ring_df["1000"] = olddf.loc[:, "dheart"]
            chest_ring_df["1000"] = olddf.loc[:, "dchest"]
            blood_ring_df["1000"] = olddf.loc[:, "dblood"]

            heart_lattice_df["1000"] = olddf.loc[:, "mlheart"]
            chest_lattice_df["1000"] = olddf.loc[:, "mlchest"]
            blood_lattice_df["1000"] = olddf.loc[:, "mlblood"]

            heart_bucket_df["1000"] = olddf.loc[:, "bheart"]
            chest_bucket_df["1000"] = olddf.loc[:, "bchest"]
            blood_bucket_df["1000"] = olddf.loc[:, "bblood"]

            heart_mixed_df["1000"] = olddf.loc[:, "mheart"]
            chest_mixed_df["1000"] = olddf.loc[:, "mchest"]
            blood_mixed_df["1000"] = olddf.loc[:, "mblood"]

        elif "_1250" in f:
            olddf = pd.read_csv(f)
            heart_ring_df["1250"] = olddf.loc[:, "dheart"]
            chest_ring_df["1250"] = olddf.loc[:, "dchest"]
            blood_ring_df["1250"] = olddf.loc[:, "dblood"]

            heart_lattice_df["1250"] = olddf.loc[:, "mlheart"]
            chest_lattice_df["1250"] = olddf.loc[:, "mlchest"]
            blood_lattice_df["1250"] = olddf.loc[:, "mlblood"]

            heart_bucket_df["1250"] = olddf.loc[:, "bheart"]
            chest_bucket_df["1250"] = olddf.loc[:, "bchest"]
            blood_bucket_df["1250"] = olddf.loc[:, "bblood"]

            heart_mixed_df["1250"] = olddf.loc[:, "mheart"]
            chest_mixed_df["1250"] = olddf.loc[:, "mchest"]
            blood_mixed_df["1250"] = olddf.loc[:, "mblood"]

        elif "_1500" in f:
            olddf = pd.read_csv(f)
            heart_ring_df["1500"] = olddf.loc[:, "dheart"]
            chest_ring_df["1500"] = olddf.loc[:, "dchest"]
            blood_ring_df["1500"] = olddf.loc[:, "dblood"]

            heart_lattice_df["1500"] = olddf.loc[:, "mlheart"]
            chest_lattice_df["1500"] = olddf.loc[:, "mlchest"]
            blood_lattice_df["1500"] = olddf.loc[:, "mlblood"]

            heart_bucket_df["1500"] = olddf.loc[:, "bheart"]
            chest_bucket_df["1500"] = olddf.loc[:, "bchest"]
            blood_bucket_df["1500"] = olddf.loc[:, "bblood"]

            heart_mixed_df["1500"] = olddf.loc[:, "mheart"]
            chest_mixed_df["1500"] = olddf.loc[:, "mchest"]
            blood_mixed_df["1500"] = olddf.loc[:, "mblood"]

        elif "_1750" in f:
            olddf = pd.read_csv(f)
            heart_ring_df["1750"] = olddf.loc[:, "dheart"]
            chest_ring_df["1750"] = olddf.loc[:, "dchest"]
            blood_ring_df["1750"] = olddf.loc[:, "dblood"]

            heart_lattice_df["1750"] = olddf.loc[:, "mlheart"]
            chest_lattice_df["1750"] = olddf.loc[:, "mlchest"]
            blood_lattice_df["1750"] = olddf.loc[:, "mlblood"]

            heart_bucket_df["1750"] = olddf.loc[:, "bheart"]
            chest_bucket_df["1750"] = olddf.loc[:, "bchest"]
            blood_bucket_df["1750"] = olddf.loc[:, "bblood"]

            heart_mixed_df["1750"] = olddf.loc[:, "mheart"]
            chest_mixed_df["1750"] = olddf.loc[:, "mchest"]
            blood_mixed_df["1750"] = olddf.loc[:, "mblood"]
        
        elif "_2000" in f:
            olddf = pd.read_csv(f)
            heart_ring_df["2000"] = olddf.loc[:, "dheart"]
            chest_ring_df["2000"] = olddf.loc[:, "dchest"]
            blood_ring_df["2000"] = olddf.loc[:, "dblood"]

            heart_lattice_df["2000"] = olddf.loc[:, "mlheart"]
            chest_lattice_df["2000"] = olddf.loc[:, "mlchest"]
            blood_lattice_df["2000"] = olddf.loc[:, "mlblood"]

            heart_bucket_df["2000"] = olddf.loc[:, "bheart"]
            chest_bucket_df["2000"] = olddf.loc[:, "bchest"]
            blood_bucket_df["2000"] = olddf.loc[:, "bblood"]

            heart_mixed_df["2000"] = olddf.loc[:, "mheart"]
            chest_mixed_df["2000"] = olddf.loc[:, "mchest"]
            blood_mixed_df["2000"] = olddf.loc[:, "mblood"]

        sns.boxplot(data=heart_ring_df, ax=heartax[0], notch=True, width=0.2, linewidth=0.2, fliersize=0.1)
        sns.boxplot(data=heart_lattice_df, ax=heartax[1], notch=True, width=0.2, linewidth=0.2, fliersize=0.1)
        sns.boxplot(data=heart_bucket_df, ax=heartax[2], notch=True, width=0.2, linewidth=0.2, fliersize=0.1)
        sns.boxplot(data=heart_mixed_df, ax=heartax[3], notch=True, width=0.2, linewidth=0.2, fliersize=0.1)

        sns.boxplot(data=chest_ring_df, ax=chestax[0], notch=True, width=0.2, linewidth=0.2, fliersize=0.1)
        sns.boxplot(data=chest_lattice_df, ax=chestax[1], notch=True, width=0.2, linewidth=0.2, fliersize=0.1)
        sns.boxplot(data=chest_bucket_df, ax=chestax[2], notch=True, width=0.2, linewidth=0.2, fliersize=0.1)
        sns.boxplot(data=chest_mixed_df, ax=chestax[3], notch=True, width=0.2, linewidth=0.2, fliersize=0.1)

        sns.boxplot(data=blood_ring_df, ax=bloodax[0], notch=True, width=0.2, linewidth=0.2, fliersize=0.1)
        sns.boxplot(data=blood_lattice_df, ax=bloodax[1], notch=True, width=0.2, linewidth=0.2, fliersize=0.1)
        sns.boxplot(data=blood_bucket_df, ax=bloodax[2], notch=True, width=0.2, linewidth=0.2, fliersize=0.1)
        sns.boxplot(data=blood_mixed_df, ax=bloodax[3], notch=True, width=0.2, linewidth=0.2, fliersize=0.1)

        heartfig.suptitle("heart training datalengths")
        heartfig.savefig("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/heart_training_lengths.png", format="png")
        
        chestfig.suptitle("chest training datalengths")
        chestfig.savefig("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/chest_training_lengths.png", format="png")
        
        bloodfig.suptitle("blood training datalengths")
        bloodfig.savefig("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/blood_training_lengths.png", format="png")

def plot_material_mixing_by_dataset():
    single_input = pd.read_csv("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/sleep_apnea_separated_input.csv")
    unspec_df = pd.read_csv("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/sleep_apnea_unspecified_input.csv")
    sel_unspec_df = pd.read_csv("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/sleep_apnea_selected_material_unspecified_input.csv")
    sel_matched_df = pd.read_csv("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/sleep_apnea_selected_material_matched_input.csv")

    fig, axs, = plt.subplots(3, 1, figsize=(3 * 5, 9))  
    # chestfig, chestax = plt.subplots(1, 1, figsize=(3 * 4, 6))  
    # bloodfig, bloodax = plt.subplots(1, 1, figsize=(3 * 4, 6))  

    sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})    
    heartdf = pd.DataFrame(columns=["(i) bucket", "(ii) bucket", "(iii)", "(iv)", "(v)"])
    chestdf = pd.DataFrame(columns=["(i) lattice", "(ii) lattice", "(iii)", "(iv)", "(v)"])
    blooddf = pd.DataFrame(columns=["(i) lattice", "(ii) lattice", "(iii)", "(iv)", "(v)"])

    mat_mixing_pal = {
        "(i) lattice": "#E69F00",
        "(i) bucket": "#E69F00",
        "(ii) lattice": "#D55E00",
        "(ii) bucket": "#D55E00",
        "(iii)": "#009E73", 
        "(iv)": "#0072B2", 
        "(v)": "#56B4E9"
    }

    heartdf["(i) bucket"] = single_input.loc[:, "bheart"]
    heartdf["(ii) bucket"] = unspec_df.loc[:, "bheart"]
    heartdf["(iii)"] = unspec_df.loc[:, "mheart"]
    heartdf["(iv)"] = sel_unspec_df.loc[:, "heart"]
    heartdf["(v)"] = sel_matched_df.loc[:, "heart"]
    
    sns.boxplot(data=heartdf, ax=axs[0], notch=True, width=0.5, linewidth=1, fliersize=3, palette=mat_mixing_pal)
    axs[0].set_ylabel('NRMSE')
    axs[0].set_xlabel('Heart Rate')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[0].set_ylim(0, 0.6)

    # heartfig.tight_layout(pad=1)
    # # heartfig.suptitle("Heart Rate Prediction with Mixed Materials")
    # heartfig.savefig("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/mixed_materials_heart.png", format="png")
    
    chestdf["(i) lattice"] = single_input.loc[:, "mlchest"]
    chestdf["(ii) lattice"] = unspec_df.loc[:, "mlchest"]
    chestdf["(iii)"] = unspec_df.loc[:, "mchest"]
    chestdf["(iv)"] = sel_unspec_df.loc[:, "chest"]
    chestdf["(v)"] = sel_matched_df.loc[:, "chest"]
    
    sns.boxplot(data=chestdf, ax=axs[1], notch=True, width=0.5, linewidth=1, fliersize=3, palette=mat_mixing_pal)
    axs[1].set_ylabel('NRMSE')
    axs[1].set_xlabel('Respiration')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].set_ylim(0, 1)

    # chestfig.tight_layout(pad=1)
    # # chestfig.suptitle("Respiration Prediction with Mixed Materials")
    # chestfig.savefig("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/mixed_materials_respiration.png", format="png")
    
    blooddf["(i) lattice"] = single_input.loc[:, "mlblood"]
    blooddf["(ii) lattice"] = unspec_df.loc[:, "mlblood"]
    blooddf["(iii)"] = unspec_df.loc[:, "mblood"]
    blooddf["(iv)"] = sel_unspec_df.loc[:, "blood"]
    blooddf["(v)"] = sel_matched_df.loc[:, "blood"]
    
    sns.boxplot(data=blooddf, ax=axs[2], notch=True, width=0.5, linewidth=1, fliersize=3, palette=mat_mixing_pal)
    axs[2].set_ylabel('NRMSE')
    axs[2].set_xlabel('Blood Oxygen Saturation')
    axs[2].spines['right'].set_visible(False)
    axs[2].spines['top'].set_visible(False)
    axs[2].set_ylim(0, 0.25)

    fig.tight_layout(pad=1)
    # bloodfig.suptitle("Blood Oxygen Saturation Prediction with Mixed Materials")
    fig.savefig("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/mixed_materials_all.png", format="png")
    return        

def plot_conclusion_results():
    unspec_df = pd.read_csv("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/sleep_apnea_unspecified_input.csv")
    single_matched_df = pd.read_csv("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/sleep_apnea_single_material_matched_input.csv")
    sel_unspec_df = pd.read_csv("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/sleep_apnea_selected_material_unspecified_input.csv")
    sel_matched_df = pd.read_csv("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/sleep_apnea_selected_material_matched_input.csv")

      
    bloodfig, bloodax = plt.subplots(1, 1, figsize=(3 * 4, 5))  

    sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})    
    
    blooddf = pd.DataFrame(columns=["(ii) lattice", "(ii*) lattice", "(iv)", "(v)"])

    mat_mixing_pal = {
        "(ii) lattice": "#D55E00",
        "(ii*) lattice" : "#D55E00",
        "(iv)": "#0072B2", 
        "(v)": "#56B4E9"
    }

    blooddf["(ii) lattice"] = unspec_df.loc[:, "mlblood"]
    blooddf["(ii*) lattice"] = single_matched_df.loc[:, "blood"]
    blooddf["(iv)"] = sel_unspec_df.loc[:, "blood"]
    blooddf["(v)"] = sel_matched_df.loc[:, "blood"]
    
    sns.boxplot(data=blooddf, ax=bloodax, notch=True, width=0.5, linewidth=1, fliersize=3, palette=mat_mixing_pal)
    bloodax.set_ylabel('NRMSE')
    bloodax.set_xlabel('Blood Oxygen Saturation')
    bloodax.spines['right'].set_visible(False)
    bloodax.spines['top'].set_visible(False)
    bloodax.set_ylim(0, 0.60)

    bloodfig.tight_layout(pad=2)
    bloodfig.suptitle("Blood Oxygen Saturation Input Mapping with Single and Selected Materials")
    bloodfig.savefig("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/input_mapping_blood.png", format="png")
    return
    return
process_results()
# plot_training_lengths()
# sleep_apnea_separately("/home/cw1647/phd/fakemat_experiments/paper_sleep_apnea/results_chopped/sleep_apnea_separated_input.csv", "separated inputs")
plot_material_mixing_by_dataset()
plot_conclusion_results()