import pandas as pd
from pathlib import Path
import numpy as np

def process_results(path):
    print(path)
    filenames =[str(filepath) for filepath in Path(path).glob('*') if ".csv" in str(filepath)]
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
            i=0
            pass
        pieces = file.split("results")
        new_filename = pieces[0]+"results_chopped"+pieces[1]
        pieces_plus = pieces[1].split(".")
        proportions = f"{pieces[0]}results_chopped{pieces_plus[0]}_proportions.{pieces_plus[1]}"
        print(new_filename)
        newdf.to_csv(new_filename)  
        chopped_prop.to_csv(proportions)
    return

# process_results("/home/cw1647/phd/het_reservoir_experiments/sleep_apnea/results")
process_results("/home/cw1647/phd/het_reservoir_experiments/mso/results")