import seaborn as sns
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

my_pal = {'bucket': 'blue',
          'bblood' : 'blue',
          'bchest' : 'blue',
          'bheart': 'blue',
          'b2': 'blue',
          'b4': 'blue',
          'b8': 'blue',
          'delayline': 'green',
          'dblood' : 'green',
          'dchest' : 'green',
          'dheart' : 'green',
          'd2' : 'green',
          'd4' : 'green',
          'd8' : 'green',
          'maglattice': 'gold',
          'mlblood':'gold',
          'mlchest':'gold',
          'mlheart':'gold',
          'ml2' : 'gold',
          'ml4' : 'gold',
          'ml8' : 'gold',
          'mixed':'darkorange',
          'mblood': 'darkorange',
          'mchest': 'darkorange',
          'mheart': 'darkorange',
          'm2' : 'darkorange',
          'm4' : 'darkorange',
          'm8' : 'darkorange'}

sns.set_context(rc={'font.size': 18, 'axes.titlesize': 28, 'axes.labelsize': 28})

def mso(path, low, high, title):
    errordf = pd.DataFrame(columns=['b2', 'b4', 'b8', 'd2', 'd4', 'd8', 'ml2', 'ml4', 'ml8', 'm2', 'm4', 'm8'])
    filenames = [str(path) for path in Path(path).glob('**/*') if "mso" in str(path)]
    for f in filenames:
        if "eight" in f:
            print(f)
            eightdf = pd.read_csv(f)
            errordf["b8"] = eightdf.loc[:, "bucket"].values
            errordf["d8"] = eightdf.loc[:, "delayline"].values
            errordf["ml8"] = eightdf.loc[:, "maglattice"].values
            errordf["m8"] = eightdf.loc[:, "mixed"].values
        elif "four" in f:
            print(f)
            fourdf = pd.read_csv(f)
            errordf["b4"] = fourdf.loc[:, "bucket"].values
            errordf["d4"] = fourdf.loc[:, "delayline"].values
            errordf["ml4"] = fourdf.loc[:, "maglattice"].values
            errordf["m4"] = fourdf.loc[:, "mixed"].values
        elif "two" in f:
            print(f)
            twodf = pd.read_csv(f)
            errordf["b2"] = twodf.loc[:, "bucket"].values
            errordf["d2"] = twodf.loc[:, "delayline"].values
            errordf["ml2"] = twodf.loc[:, "maglattice"].values
            errordf["m2"] = twodf.loc[:, "mixed"].values
    fig, ax = plt.subplots(1, 1, figsize=(4 * 4, 5))
    sns.set_context(rc={'font.size': 28, 'axes.titlesize': 28, 'axes.labelsize': 28})
    sns.boxplot(data=errordf, ax=ax, notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    ax.set_ylabel('NRMSE')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(low, high)
    fig.suptitle(title)
    fig.savefig(path+"/MSO.png", format="png")
    return

def mso_alternate(path, low, high, title):
    errordf = pd.DataFrame(columns=["b2", "d2", "ml2", "m2", "b4", "d4", "ml4", "m4", "b8", "d8", "ml8", "m8"])
    filenames = [str(path) for path in Path(path).glob('**/*') if "mso" in str(path)]
    for f in filenames:
        if "eight" in f:
            print(f)
            eightdf = pd.read_csv(f)
            errordf["b8"] = eightdf.loc[:, "bucket"].values
            errordf["d8"] = eightdf.loc[:, "delayline"].values
            errordf["ml8"] = eightdf.loc[:, "maglattice"].values
            errordf["m8"] = eightdf.loc[:, "mixed"].values
        elif "four" in f:
            print(f)
            fourdf = pd.read_csv(f)
            errordf["b4"] = fourdf.loc[:, "bucket"].values
            errordf["d4"] = fourdf.loc[:, "delayline"].values
            errordf["ml4"] = fourdf.loc[:, "maglattice"].values
            errordf["m4"] = fourdf.loc[:, "mixed"].values
        elif "two" in f:
            print(f)
            twodf = pd.read_csv(f)
            errordf["b2"] = twodf.loc[:, "bucket"].values
            errordf["d2"] = twodf.loc[:, "delayline"].values
            errordf["ml2"] = twodf.loc[:, "maglattice"].values
            errordf["m2"] = twodf.loc[:, "mixed"].values
    fig, ax = plt.subplots(1, 1, figsize=(4 * 4, 5))
    sns.set_context(rc={'font.size': 28, 'axes.titlesize': 28, 'axes.labelsize': 28})
    sns.boxplot(data=errordf, ax=ax, notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    ax.set_ylabel('NRMSE')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(low, high)
    fig.suptitle(title)
    fig.savefig(path+"/MSO_alternate.png", format="png")

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

def mso_separately(path, title):
    filenames = [str(path) for path in Path(path).glob('**/*') if "mso" in str(path) and "csv" in str(path)]
    fig, ax = plt.subplots(1, 3, figsize=(6* 4, 6))
    sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})    
    for f in filenames:
        if "eight" in f:
            eightdf = pd.read_csv(f)
            low = max(min(eightdf.loc[:, "bucket"].min()-0.1, eightdf.loc[:, "delayline"].min()-0.1, eightdf.loc[:, "maglattice"].min()-0.1, eightdf.loc[:, "mixed"].min()-0.1), 0)
            high = min(max(eightdf.loc[:, "bucket"].max()+0.1, eightdf.loc[:, "delayline"].max()+0.1, eightdf.loc[:, "maglattice"].max()+0.1, eightdf.loc[:, "mixed"].max()+0.1), 1)
            sns.boxplot(data=eightdf, ax=ax[2], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
            ax[2].set_ylabel('NRMSE')
            ax[2].set_xlabel('MSO 8')
            ax[2].spines['right'].set_visible(False)
            ax[2].spines['top'].set_visible(False)
            ax[2].set_ylim(low, high)
        elif "four" in f:
            fourdf = pd.read_csv(f)
            low = max(min(fourdf.loc[:, "bucket"].min()-0.1, fourdf.loc[:, "delayline"].min()-0.1, fourdf.loc[:, "maglattice"].min()-0.1, fourdf.loc[:, "mixed"].min()-0.1), 0)
            high = min(max(fourdf.loc[:, "bucket"].max()+0.1, fourdf.loc[:, "delayline"].max()+0.1, fourdf.loc[:, "maglattice"].max()+0.1, fourdf.loc[:, "mixed"].max()+0.1), 1)
            sns.boxplot(data=fourdf, ax=ax[1], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
            ax[1].set_ylabel('NRMSE')
            ax[1].set_xlabel('MSO 4')
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['top'].set_visible(False)
            ax[1].set_ylim(low, high)
        else:
            twodf = pd.read_csv(f)
            low = max(min(twodf.loc[:, "bucket"].min()-0.1, twodf.loc[:, "delayline"].min()-0.1, twodf.loc[:, "maglattice"].min()-0.1, twodf.loc[:, "mixed"].min()-0.1), 0)
            high = min(max(twodf.loc[:, "bucket"].max()+0.1, twodf.loc[:, "delayline"].max()+0.1, twodf.loc[:, "maglattice"].max()+0.1, twodf.loc[:, "mixed"].max()+0.1), 1)
            sns.boxplot(data=twodf, ax=ax[0], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
            ax[0].set_ylabel('NRMSE')
            ax[0].set_xlabel('MSO 2')
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['top'].set_visible(False)
            ax[0].set_ylim(low, high)
    fig.suptitle(title)
    fig.savefig(path+"/MSO_separate.png", format="png")
    return

def sleep_apnea_separately(path, title):
    errordf = pd.read_csv(path)
    fig, ax = plt.subplots(1, 3, figsize=(6 * 4, 6))  
    sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})    
    heartdf = pd.DataFrame(columns=["bucket", "delayline", "maglattice", "mixed"])
    chestdf = pd.DataFrame(columns=["bucket", "delayline", "maglattice", "mixed"])
    blooddf = pd.DataFrame(columns=["bucket", "delayline", "maglattice", "mixed"])

    heartdf["bucket"] = errordf.loc[:, "bheart"]
    heartdf["delayline"] = errordf.loc[:, "dheart"]
    heartdf["maglattice"] = errordf.loc[:, "mlheart"]
    heartdf["mixed"] = errordf.loc[:, "mheart"]
    low = max(min(heartdf.loc[:, "bucket"].min()-0.1, heartdf.loc[:, "delayline"].min()-0.1, heartdf.loc[:, "maglattice"].min()-0.1, heartdf.loc[:, "mixed"].min()-0.1), 0)
    high = min(max(heartdf.loc[:, "bucket"].max()+0.1, heartdf.loc[:, "delayline"].max()+0.1, heartdf.loc[:, "maglattice"].max()+0.1, heartdf.loc[:, "mixed"].max()+0.1), 1)
    sns.boxplot(data=heartdf, ax=ax[0], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    ax[0].set_ylabel('NRMSE')
    ax[0].set_xlabel('heart rate')
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].set_ylim(low, high)

    chestdf["bucket"] = errordf.loc[:, "bchest"]
    chestdf["delayline"] = errordf.loc[:, "dchest"]
    chestdf["maglattice"] = errordf.loc[:, "mlchest"]
    chestdf["mixed"] = errordf.loc[:, "mchest"]
    low = max(min(chestdf.loc[:, "bucket"].min()-0.1, chestdf.loc[:, "delayline"].min()-0.1, chestdf.loc[:, "maglattice"].min()-0.1, chestdf.loc[:, "mixed"].min()-0.1), 0)
    high = min(max(chestdf.loc[:, "bucket"].max()+0.1, chestdf.loc[:, "delayline"].max()+0.1, chestdf.loc[:, "maglattice"].max()+0.1, chestdf.loc[:, "mixed"].max()+0.1), 1)
    sns.boxplot(data=chestdf, ax=ax[1], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    ax[1].set_ylabel('NRMSE')
    ax[1].set_xlabel('chest volume')
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_ylim(low, high)

    blooddf["bucket"] = errordf.loc[:, "bblood"]
    blooddf["delayline"] = errordf.loc[:, "dblood"]
    blooddf["maglattice"] = errordf.loc[:, "mlblood"]
    blooddf["mixed"] = errordf.loc[:, "mblood"]
    low = max(min(blooddf.loc[:, "bucket"].min()-0.1, blooddf.loc[:, "delayline"].min()-0.1, blooddf.loc[:, "maglattice"].min()-0.1, blooddf.loc[:, "mixed"].min()-0.1), 0)
    high = min(max(blooddf.loc[:, "bucket"].max()+0.1, blooddf.loc[:, "delayline"].max()+0.1, blooddf.loc[:, "maglattice"].max()+0.1, blooddf.loc[:, "mixed"].max()+0.1), 1)
    sns.boxplot(data=blooddf, ax=ax[2], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    ax[2].set_ylabel('NRMSE')
    ax[2].set_xlabel('blood oxygenation')
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['top'].set_visible(False)
    ax[2].set_ylim(low, high)

    fig.suptitle(title)
    fig.savefig(path.split(".")[0]+"_separate.png", format="png")
    return

mso("/home/cw1647/phd/fakemat_experiments/results_chopped/multi_timescale", 0, 0.6, "MSO multiple timescales")
# mso("/home/cw1647/phd/fakemat_experiments/results_chopped/single_timescale", 0, 1, "MSO single timescales")
mso("/home/cw1647/phd/fakemat_experiments/results_chopped/normalised_total_svd/multi_timescale", 0, 0.6, "MSO multiple timescales & svd normalisation")
# mso("/home/cw1647/phd/fakemat_experiments/results_chopped/normalised_total_svd/single_timescale", 0, 0.2, "MSO single timescale & svd normalisation")

mso_alternate("/home/cw1647/phd/fakemat_experiments/results_chopped/multi_timescale", 0, 0.6, "MSO multiple timescales alternate")
# mso_alternate("/home/cw1647/phd/fakemat_experiments/results_chopped/single_timescale", 0, 1, "MSO single timescales alternate")
mso_alternate("/home/cw1647/phd/fakemat_experiments/results_chopped/normalised_total_svd/multi_timescale", 0, 0.6, "MSO multiple timescales & svd normalisation alternate")
# mso_alternate("/home/cw1647/phd/fakemat_experiments/results_chopped/normalised_total_svd/single_timescale", 0, 0.2, "MSO single timescale & svd normalisation alternate")

mso_separately("/home/cw1647/phd/fakemat_experiments/results_chopped/multi_timescale", "MSO multiple timescales separated")
# mso_separately("/home/cw1647/phd/fakemat_experiments/results_chopped/single_timescale",  "MSO single timescales separated")
mso_separately("/home/cw1647/phd/fakemat_experiments/results_chopped/normalised_total_svd/multi_timescale", "MSO multiple timescales & svd normalisation separated")
# mso_separately("/home/cw1647/phd/fakemat_experiments/results_chopped/normalised_total_svd/single_timescale", "MSO single timescale & svd normalisation separated")


sleep_apnea("fakemat_experiments/results_chopped/multi_timescale/sleep_apnea.csv", 0, 1, "Sleep Apnea multiple timescales")
# sleep_apnea("fakemat_experiments/results_chopped/single_timescale/sleep_apnea.csv", 0, 1, "Sleep Apnea single timescale")
sleep_apnea("/home/cw1647/phd/fakemat_experiments/results_chopped/normalised_total_svd/multi_timescale/sleep_apnea.csv", 0, 1, "Sleep Apnea multiple timescales & normalised SVD")
# sleep_apnea("/home/cw1647/phd/fakemat_experiments/results_chopped/normalised_total_svd/single_timescale/sleep_apnea.csv", 0, 1, "Sleep Apnea single timescale & normalised SVD")

sleep_apnea_alternate("fakemat_experiments/results_chopped/multi_timescale/sleep_apnea.csv", 0, 1, "Sleep Apnea multiple timescales alternate")
# sleep_apnea_alternate("fakemat_experiments/results_chopped/single_timescale/sleep_apnea.csv", 0, 1, "Sleep Apnea single timescale alternate")
sleep_apnea_alternate("/home/cw1647/phd/fakemat_experiments/results_chopped/normalised_total_svd/multi_timescale/sleep_apnea.csv", 0, 1, "Sleep Apnea multiple timescales & normalised SVD alternate")
# sleep_apnea_alternate("/home/cw1647/phd/fakemat_experiments/results_chopped/normalised_total_svd/single_timescale/sleep_apnea.csv", 0, 1, "Sleep Apnea single timescale & normalised SVD alternate")

sleep_apnea_separately("fakemat_experiments/results_chopped/multi_timescale/sleep_apnea.csv", "Sleep Apnea multiple timescales separated")
# sleep_apnea_separately("fakemat_experiments/results_chopped/single_timescale/sleep_apnea.csv", "Sleep Apnea single timescale separated")
sleep_apnea_separately("/home/cw1647/phd/fakemat_experiments/results_chopped/normalised_total_svd/multi_timescale/sleep_apnea.csv", "Sleep Apnea multiple timescales & normalised SVD separated")
# sleep_apnea_separately("/home/cw1647/phd/fakemat_experiments/results_chopped/normalised_total_svd/single_timescale/sleep_apnea.csv", "Sleep Apnea single timescale & normalised SVD separated")