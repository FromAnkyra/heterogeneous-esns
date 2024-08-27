from pathlib import Path
from NymphESN import vis
import pandas as pd
import glob
from functools import reduce

def boxplot_csv(filename):
    df = pd.read_csv(filename, index_col=0)
    df = df.filter(like="test")
    # print(f"{df.min().min()=}")
    # print(f"total min: {min([df[idx] for idx in df.idxmin()])}")
    buf = filename.split('.')[0] + '.png'
    # print(f"{filename}: [{df.min().min()}, {df.max().max()}]")
    title = gen_title(filename)
    vis.ErrorVis.vis(df, buf, df.min().min(), df.max().mean(), title=title)

def gen_title(filename):
    words = filename.split('/')
    words = words[words.index("tempESN-experiments")+1:words.index("results")] + [words[-1].split(".")[0]]
    return reduce(lambda x, y: x+" "+y, words)

filenames = glob.glob("/home/cw1647/phd/tempESN-experiments/**/**/**/**.csv", recursive=True)
# print(f"{filenames=}")
filenames = set(filenames)
for f in filenames:
    boxplot_csv(f)