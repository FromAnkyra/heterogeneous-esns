import pandas as pd
import numpy as np
from pathlib import Path

filenames =[str(path) for path in Path("/home/cw1647/phd/fakemat_experiments/results").glob('**/*') if ".csv" in str(path)]
# print(filenames)

def f(item):
    if item > 1:
        return 1.0
    else: 
        return item
vf = np.vectorize(f)


for file in filenames:
    print(file)
    df = pd.read_csv(file)
    newdf = pd.DataFrame(columns=list(df.columns[1:]))

    for col in df.columns[1:]:
        data = df.loc[:, col].values
        newdf[col] = vf(data)
        pass
    pieces = file.split("results")
    new_filename = pieces[0]+"results_chopped"+pieces[1]
    print(new_filename)
    newdf.to_csv(new_filename)


    