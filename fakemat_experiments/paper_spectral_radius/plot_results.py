import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

my_pal = {
          'b2_scaled': '#b60a3b',
          'b4_scaled': '#b60a3b',
          'b8_scaled': '#b60a3b',
          'b2': '#db7476',
          'b4': '#db7476',
          'b8': '#db7476',
          'r2_scaled' : '#d45200',
          'r4_scaled' : '#d45200',
          'r8_scaled' : '#d45200',
          'r2' : '#ed8c37',
          'r4' : '#ed8c37',
          'r8' : '#ed8c37',
          'rs2_scaled' : '#fbb430',
          'rs4_scaled' : '#fbb430',
          'rs8_scaled' : '#fbb430',
          'rs2' : '#f5a936',
          'rs4' : '#f5a936',
          'rs8' : '#f5a936',
          'ml2_scaled' : '#5a7356',
          'ml4_scaled' : '#5a7356',
          'ml8_scaled' : '#5a7356',
          'ml2' : '#51a885',
          'ml4' : '#51a885',
          'ml8' : '#51a885',
          'm2_scaled' : '#006b8f',
          'm4_scaled' : '#006b8f',
          'm8_scaled' : '#006b8f',
          'm2' : '#267a9e',
          'm4' : '#267a9e',
          'm8' : '#267a9e',
          'sm2_scaled' : '#7b3b66',
          'sm4_scaled' : '#7b3b66',
          'sm8_scaled' : '#7b3b66',
          'sm2' : '#986b9r',
          'sm4' : '#986b9r',
          'sm8' : '#986b9r'}

sns.set_context(rc={'font.size': 18, 'axes.titlesize': 28, 'axes.labelsize': 28})

def process_results():
    filenames =[str(path) for path in Path("/home/cw1647/phd/fakemat_experiments/paper_spectral_radius/results/").glob('*') if ".csv" in str(path)]
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
        newdf = pd.DataFrame(columns=list(df.columns[1:]))
        chopped_prop = pd.DataFrame(columns=list(df.columns[1:]))
        for col in df.columns[1:]:
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

def plot_results(path):
    errordf = pd.DataFrame(columns=['b2_scaled', 'b4_scaled', 'b8_scaled', 
                                    'b2', 'b4', 'b8',
                                    'r2_scaled', 'r4_scaled', 'r8_scaled',
                                    'r2', 'r4', 'r8', 
                                    'rs2_scaled', 'rs4_scaled', 'rs8_scaled',
                                    'rs2', 'rs4', 'rs8',  
                                    'ml2_scaled', 'ml4_scaled', 'ml8_scaled',
                                    'ml2', 'ml4', 'ml8', 
                                    'm2_scaled', 'm4_scaled', 'm8_scaled',
                                    'm2', 'm4', 'm8', 
                                    'sm2_scaled', 'sm4_scaled', 'sm8_scaled',
                                    'sm2', 'sm4', 'sm8'])
    filenames = [str(path) for path in Path(path).glob('**/*') if "csv" in str(path) and "proportions" not in str(path)]
    # unscaled rings
    ringsdf2 = pd.DataFrame(columns=['r2', 'rs2'])
    ringsdf4 = pd.DataFrame(columns=['r4', 'rs4'])
    ringsdf8 = pd.DataFrame(columns=['r8', 'rs8'])

    latticedf2 = pd.DataFrame(columns=['ml2', 'ml2_scaled'])
    latticedf4 = pd.DataFrame(columns=['ml4', 'ml4_scaled'])
    latticedf8 = pd.DataFrame(columns=['ml8', 'ml8_scaled'])

    bucketdf2 = pd.DataFrame(columns=['b2', 'b2_scaled'])
    bucketdf4 = pd.DataFrame(columns=['b4', 'b4_scaled'])
    bucketdf8 = pd.DataFrame(columns=['b8', 'b8_scaled'])

    nonsymring2 = pd.DataFrame(columns=['r2', 'r2_scaled'])
    nonsymring4 = pd.DataFrame(columns=['r4', 'r4_scaled'])
    nonsymring8 = pd.DataFrame(columns=['r8', 'r8_scaled'])

    mixed2 = pd.DataFrame(columns=['m2', 'm2_scaled'])
    mixed4 = pd.DataFrame(columns=['m4', 'm4_scaled'])
    mixed8 = pd.DataFrame(columns=['m8', 'm8_scaled'])

    for f in filenames:
        if "eight" in f:
            if "normalised" in f:
                errordf = pd.read_csv(f)
                latticedf8['ml8_scaled'] = errordf.loc[:, "maglattice"]
                bucketdf8['b8_scaled'] = errordf.loc[:, "bucket"]
                nonsymring8['r8_scaled'] = errordf.loc[:, "delayline"]
                mixed8['m8_scaled'] = errordf.loc[:, "mixed"]
            else:
                errordf = pd.read_csv(f)
                ringsdf8['r8'] = errordf.loc[:, 'delayline']
                ringsdf8['rs8'] = errordf.loc[:, 'sdelayline']
                latticedf8['ml8'] = errordf.loc[:, "maglattice"]
                bucketdf8['b8'] = errordf.loc[:, "bucket"]
                nonsymring8['r8'] = errordf.loc[:, 'delayline']
                mixed8['m8'] = errordf.loc[:, "mixed"]
        elif "four" in f:
            if "normalised" in f:
                errordf = pd.read_csv(f)
                latticedf4['ml4_scaled'] = errordf.loc[:, "maglattice"]
                bucketdf4['b4_scaled'] = errordf.loc[:, "bucket"]
                nonsymring4['r4_scaled'] = errordf.loc[:, "delayline"]
                mixed4['m4_scaled'] = errordf.loc[:, "mixed"]
            else:
                errordf = pd.read_csv(f)
                ringsdf4['r4'] = errordf.loc[:, 'delayline']
                ringsdf4['rs4'] = errordf.loc[:, 'sdelayline']
                latticedf4['ml4'] = errordf.loc[:, 'maglattice']
                bucketdf4['b4'] = errordf.loc[:, 'bucket']
                nonsymring4['r4'] = errordf.loc[:, 'delayline']
                mixed4['m4'] = errordf.loc[:, "mixed"]

        else:
            if "normalised" in f:
                errordf = pd.read_csv(f)
                latticedf2['ml2_scaled'] = errordf.loc[:, "maglattice"]
                bucketdf2['b2_scaled'] = errordf.loc[:, "bucket"]
                nonsymring2['r2_scaled'] = errordf.loc[:, "delayline"]
                mixed2['m2_scaled'] = errordf.loc[:, "mixed"]
            else:
                errordf = pd.read_csv(f)
                ringsdf2['r2'] = errordf.loc[:, 'delayline']
                ringsdf2['rs2'] = errordf.loc[:, 'sdelayline']
                latticedf2['ml2'] = errordf.loc[:, 'maglattice']
                bucketdf2['b2'] = errordf.loc[:, 'bucket']
                nonsymring2['r2'] = errordf.loc[:, 'delayline']
                mixed2['m2'] = errordf.loc[:, "mixed"]

    ringsfig, raxs = plt.subplots(1, 3, figsize=(4 * 4, 5))
    latticefig, laxs = plt.subplots(1, 3, figsize=(4 * 4, 5))
    bucketfig, baxs = plt.subplots(1, 3, figsize=(4 * 4, 5))
    nonsymrigfig, naxs = plt.subplots(1, 3, figsize=(4 * 4, 5))
    mixedfig, maxs = plt.subplots(1, 3, figsize=(4 * 4, 5))
    
    sns.boxplot(data=ringsdf2, ax=raxs[0], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    low2 = max(min(ringsdf2.loc[:, "r2"].min()-0.001, ringsdf2.loc[:, "rs2"].min()-0.001), 0)
    high2 = min(max(ringsdf2.loc[:, "r2"].max()+0.001, ringsdf2.loc[:, "rs2"].max()+0.001), 1)
    raxs[0].set_ylim(low2, high2)
    raxs[0].set_ylabel('NRMSE')
    raxs[0].set_xlabel('MSO 2')
    raxs[0].spines['right'].set_visible(False)
    raxs[0].spines['top'].set_visible(False)
    sns.boxplot(data=ringsdf4, ax=raxs[1], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    low4 = max(min(ringsdf4.loc[:, "r4"].min()-0.001, ringsdf4.loc[:, "rs4"].min()-0.001), 0)
    high4 = min(max(ringsdf4.loc[:, "r4"].max()+0.001, ringsdf4.loc[:, "rs4"].max()+0.001), 1)
    raxs[1].set_ylim(low4, high4)
    raxs[1].set_ylabel('NRMSE')
    raxs[1].set_xlabel('MSO 4')
    raxs[1].spines['right'].set_visible(False)
    raxs[1].spines['top'].set_visible(False)

    sns.boxplot(data=ringsdf8, ax=raxs[2], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    low8 = max(min(ringsdf8.loc[:, "r8"].min()-0.001, ringsdf8.loc[:, "rs8"].min()-0.001), 0)
    high8 = min(max(ringsdf8.loc[:, "r8"].max()+0.001, ringsdf8.loc[:, "rs8"].max()+0.001), 1)
    raxs[2].set_ylim(low8, high8)
    raxs[2].set_ylabel('NRMSE')
    raxs[2].set_xlabel('MSO 8')
    raxs[2].spines['right'].set_visible(False)
    raxs[2].spines['top'].set_visible(False)

    ringsfig.suptitle("Symmetric vs Nonsymmetric Ring")
    ringsfig.savefig(path+"/symmetric_vs_nonsymmetric_ring.png", format="png")  
# boop
    sns.boxplot(data=latticedf2, ax=laxs[0], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    low2 = max(min(latticedf2.loc[:, "ml2"].min()-0.001, latticedf2.loc[:, "ml2_scaled"].min()-0.001), 0)
    high2 = min(max(latticedf2.loc[:, "ml2"].max()+0.001, latticedf2.loc[:, "ml2_scaled"].max()+0.001), 1)
    laxs[0].set_ylim(low2, high2)
    laxs[0].set_ylabel('NRMSE')
    laxs[0].set_xlabel('MSO 2')
    laxs[0].spines['right'].set_visible(False)
    laxs[0].spines['top'].set_visible(False)
    sns.boxplot(data=latticedf4, ax=laxs[1], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    low4 = max(min(latticedf4.loc[:, "ml4"].min()-0.001, latticedf4.loc[:, "ml4_scaled"].min()-0.001), 0)
    high4 = min(max(latticedf4.loc[:, "ml4"].max()+0.001, latticedf4.loc[:, "ml4_scaled"].max()+0.001), 1)
    laxs[1].set_ylim(low4, high4)
    laxs[1].set_ylabel('NRMSE')
    laxs[1].set_xlabel('MSO 4')
    laxs[1].spines['right'].set_visible(False)
    laxs[1].spines['top'].set_visible(False)

    sns.boxplot(data=latticedf8, ax=laxs[2], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    low8 = max(min(latticedf8.loc[:, "ml8"].min()-0.001, latticedf8.loc[:, "ml8_scaled"].min()-0.001), 0)
    high8 = min(max(latticedf8.loc[:, "ml8"].max()+0.001, latticedf8.loc[:, "ml8_scaled"].max()+0.001), 1)
    laxs[2].set_ylim(low8, high8)
    laxs[2].set_ylabel('NRMSE')
    laxs[2].set_xlabel('MSO 8')
    laxs[2].spines['right'].set_visible(False)
    laxs[2].spines['top'].set_visible(False)

    latticefig.suptitle("Scaled vs Nonscaled Lattice")
    latticefig.savefig(path+"/scaled_vs_nonscaled_lattice.png", format="png") 
    #boop
    sns.boxplot(data=bucketdf2, ax=baxs[0], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    low2 = max(min(bucketdf2.loc[:, "b2"].min()-0.001, bucketdf2.loc[:, "b2_scaled"].min()-0.001), 0)
    high2 = min(max(bucketdf2.loc[:, "b2"].max()+0.001, bucketdf2.loc[:, "b2_scaled"].max()+0.001), 1)
    baxs[0].set_ylim(low2, high2)
    baxs[0].set_ylabel('NRMSE')
    baxs[0].set_xlabel('MSO 2')
    baxs[0].spines['right'].set_visible(False)
    baxs[0].spines['top'].set_visible(False)
    sns.boxplot(data=bucketdf4, ax=baxs[1], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    low4 = max(min(bucketdf4.loc[:, "b4"].min()-0.001, bucketdf4.loc[:, "b4_scaled"].min()-0.001), 0)
    high4 = min(max(bucketdf4.loc[:, "b4"].max()+0.001, bucketdf4.loc[:, "b4_scaled"].max()+0.001), 1)
    baxs[1].set_ylim(low4, high4)
    baxs[1].set_ylabel('NRMSE')
    baxs[1].set_xlabel('MSO 4')
    baxs[1].spines['right'].set_visible(False)
    baxs[1].spines['top'].set_visible(False)

    sns.boxplot(data=bucketdf8, ax=baxs[2], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    low8 = max(min(bucketdf8.loc[:, "b8"].min()-0.001, bucketdf8.loc[:, "b8_scaled"].min()-0.001), 0)
    high8 = min(max(bucketdf8.loc[:, "b8"].max()+0.001, bucketdf8.loc[:, "b8_scaled"].max()+0.001), 1)
    baxs[2].set_ylim(low8, high8)
    baxs[2].set_ylabel('NRMSE')
    baxs[2].set_xlabel('MSO 8')
    baxs[2].spines['right'].set_visible(False)
    baxs[2].spines['top'].set_visible(False)

    bucketfig.suptitle("Scaled vs Nonscaled Bucket")
    bucketfig.savefig(path+"/scaled_vs_nonscaled_bucket.png", format="png") 
# boop 
    sns.boxplot(data=nonsymring2, ax=naxs[0], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    low2 = max(min(nonsymring2.loc[:, "r2"].min()-0.001, nonsymring2.loc[:, "r2_scaled"].min()-0.001), 0)
    high2 = min(max(nonsymring2.loc[:, "r2"].max()+0.001, nonsymring2.loc[:, "r2_scaled"].max()+0.001), 1)
    naxs[0].set_ylim(low2, high2)
    naxs[0].set_ylabel('NRMSE')
    naxs[0].set_xlabel('MSO 2')
    naxs[0].spines['right'].set_visible(False)
    naxs[0].spines['top'].set_visible(False)
    sns.boxplot(data=nonsymring4, ax=naxs[1], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    low4 = max(min(nonsymring4.loc[:, "r4"].min()-0.001, nonsymring4.loc[:, "r4_scaled"].min()-0.001), 0)
    high4 = min(max(nonsymring4.loc[:, "r4"].max()+0.001, nonsymring4.loc[:, "r4_scaled"].max()+0.001), 1)
    naxs[1].set_ylim(low4, high4)
    naxs[1].set_ylabel('NRMSE')
    naxs[1].set_xlabel('MSO 4')
    naxs[1].spines['right'].set_visible(False)
    naxs[1].spines['top'].set_visible(False)

    sns.boxplot(data=nonsymring8, ax=naxs[2], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    low8 = max(min(nonsymring8.loc[:, "r8"].min()-0.001, nonsymring8.loc[:, "r8_scaled"].min()-0.001), 0)
    high8 = min(max(nonsymring8.loc[:, "r8"].max()+0.001, nonsymring8.loc[:, "r8_scaled"].max()+0.001), 1)
    naxs[2].set_ylim(low8, high8)
    naxs[2].set_ylabel('NRMSE')
    naxs[2].set_xlabel('MSO 8')
    naxs[2].spines['right'].set_visible(False)
    naxs[2].spines['top'].set_visible(False)

    nonsymrigfig.suptitle("Scaled vs Nonscaled Ring")
    nonsymrigfig.savefig(path+"/scaled_vs_nonscaled_ring.png", format="png")

    #boop
    sns.boxplot(data=mixed2, ax=maxs[0], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    low2 = max(min(mixed2.loc[:, "m2_scaled"].min()-0.001, mixed2.loc[:, "m2"].min()-0.001), 0)
    high2 = min(max(mixed2.loc[:, "m2_scaled"].max()+0.001, mixed2.loc[:, "m2"].max()+0.001), 1)
    maxs[0].set_ylim(low2, high2)
    maxs[0].set_ylabel('NRMSE')
    maxs[0].set_xlabel('MSO 2')
    maxs[0].spines['right'].set_visible(False)
    maxs[0].spines['top'].set_visible(False)
    sns.boxplot(data=mixed4, ax=maxs[1], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    low4 = max(min(mixed4.loc[:, "m4_scaled"].min()-0.001, mixed4.loc[:, "m4"].min()-0.001), 0)
    high4 = min(max(mixed4.loc[:, "m4_scaled"].max()+0.001, mixed4.loc[:, "m4"].max()+0.001), 1)
    maxs[1].set_ylim(low4, high4)
    maxs[1].set_ylabel('NRMSE')
    maxs[1].set_xlabel('MSO 4')
    maxs[1].spines['right'].set_visible(False)
    maxs[1].spines['top'].set_visible(False)

    sns.boxplot(data=mixed8, ax=maxs[2], notch=True, width=0.2, linewidth=0.2, fliersize=0.1, palette=my_pal)
    low8 = max(min(mixed8.loc[:, "m8_scaled"].min()-0.001, mixed8.loc[:, "m8"].min()-0.001), 0)
    high8 = min(max(mixed8.loc[:, "m8_scaled"].max()+0.001, mixed8.loc[:, "m8"].max()+0.001), 1)
    maxs[2].set_ylim(low8, high8)
    maxs[2].set_ylabel('NRMSE')
    maxs[2].set_xlabel('MSO 8')
    maxs[2].spines['right'].set_visible(False)
    maxs[2].spines['top'].set_visible(False)

    mixedfig.suptitle("Mixed Reservoir with vs without scaling")
    mixedfig.savefig(path+"/mixed_scaled_vs_unscaled.png", format="png")

    return

process_results()

plot_results("/home/cw1647/phd/fakemat_experiments/paper_spectral_radius/results_chopped")
