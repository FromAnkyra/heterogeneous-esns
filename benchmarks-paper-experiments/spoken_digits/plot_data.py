import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

def plot(dir):
    print("hi")
    filenames = [str(path) for path in Path(dir).glob('*') if "csv" in str(path)]
    print(len(filenames))
    for file in filenames[0:10]:
        print(file)
        df = pd.read_csv(file)
        plt.plot(df.iloc[:,-3:])
        name = file.split(".")[0]
        print(f"{name}.png")
        plt.savefig(f"{name}.png")
        plt.clf()
    return

plot("/home/cw1647/phd/benchmarks/spoken_digits/ti46/AllTestData")