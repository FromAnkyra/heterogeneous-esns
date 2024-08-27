from pathlib import Path
from NymphESN import vis
import pandas as pd
import numpy as np



def make_plot(size, directory, low, high):
    print(f"{directory=}")
    filenames = [str(path) for path in Path(directory).glob('**/*')]
    filenames_ = [f for f in filenames if "size-" + str(size) in f or "size"+str(size) in f]
    s = []
    r2 = []
    r4 = []
    r8 = []
    print(f"{len(s)=}, {len(r2)=}, {len(r4)=}, {len(r8)=}")
    print(filenames_)
    print(f"{size=}")
    for f in filenames_:
        if "subgroups-2" in f or "subgroups2" in f:
            r2_lines = open(f, "r").readlines()[1:]
            s = [float(line.split(",")[2]) for line in r2_lines]
            r2 = [float(line.split(",")[4]) for line in r2_lines]
        elif "subgroups-4" in f or "subgroups4" in f:
            r4_lines = open(f, "r").readlines()[1:]
            r4 = [float(line.split(",")[4]) for line in r4_lines]
        elif "subgroups-8" in f or "subgroups8" in f:
            r8_lines = open(f, "r").readlines()[1:]
            r8 = [float(line.split(",")[4]) for line in r8_lines]
    print(f"{len(s)=}, {len(r2)=}, {len(r4)=}, {len(r8)=}")
    d = {"1":s, "2":r2, "4":r4, "8":r8}
    df = pd.DataFrame(data=d)
    buf = directory + str(size) + ".png"
    vis.ErrorVis.vis(df, buf, low=low, high=high)


def make_plot_log(size, directory, low, high):
    print(f"{directory=}")
    filenames = [str(path) for path in Path(directory).glob('**/*')]
    filenames_ = [f for f in filenames if "size-" + str(size) in f or "size"+str(size) in f]
    s = []
    r2 = []
    r4 = []
    r8 = []
    print(f"{len(s)=}, {len(r2)=}, {len(r4)=}, {len(r8)=}")
    print(filenames_)
    print(f"{size=}")
    for f in filenames_:
        if "subgroups-2" in f or "subgroups2" in f:
            r2_lines = open(f, "r").readlines()[1:]
            s = [np.log10(float(line.split(",")[2])) for line in r2_lines]
            r2 = [np.log10(float(line.split(",")[4])) for line in r2_lines]
        elif "subgroups-4" in f or "subgroups4" in f:
            r4_lines = open(f, "r").readlines()[1:]
            r4 = [np.log10(float(line.split(",")[4])) for line in r4_lines]
        elif "subgroups-8" in f or "subgroups8" in f:
            r8_lines = open(f, "r").readlines()[1:]
            r8 = [np.log10(float(line.split(",")[4])) for line in r8_lines]
    print(f"{len(s)=}, {len(r2)=}, {len(r4)=}, {len(r8)=}")
    d = {"1":s, "2":r2, "4":r4, "8":r8}
    df = pd.DataFrame(data=d)
    buf = directory + str(size) + ".png"
    vis.ErrorVis.vis(df, buf, low=low, high=high)

for size in [64, 128, 256, 512]:
    make_plot(size, "/home/cw1647/phd/UCNC-experiments/results/UCNC_extension/narma_overall", low=0, high=0.7)
    make_plot(size, "/home/cw1647/phd/UCNC-experiments/results/UCNC_extension/narma_patch", low=0, high=0.8)
    make_plot(size, "/home/cw1647/phd/UCNC-experiments/results/UCNC_extension/sunspots_overall", low=0, high=1)
    make_plot(size, "/home/cw1647/phd/UCNC-experiments/results/UCNC_extension/sunspots_patch", low=0, high=0.8)
    # make_plot(size, "/home/cw1647/phd/UCNC-experiments/results/UCNC_extension/mso_eight_overall", low=0, high=0.3)
    # make_plot(size, "/home/cw1647/phd/UCNC-experiments/results/UCNC_extension/mso_eight_patch", low=0, high=0.2)
    # make_plot(size, "/home/cw1647/phd/UCNC-experiments/results/UCNC_extension/mso_two_overall", low=0, high=0.05)
    # make_plot(size, "/home/cw1647/phd/UCNC-experiments/results/UCNC_extension/mso_two_patch", low=0, high=0.02)
    make_plot_log(size, "/home/cw1647/phd/UCNC-experiments/extension/mso_eight_patch", low=-3, high=-1.5)
    make_plot_log(size, "/home/cw1647/phd/UCNC-experiments/extension/mso_four_patch", low=-3, high=0)
    make_plot_log(size, "/home/cw1647/phd/UCNC-experiments/extension/mso_two_patch", low=-4, high=-2.5)

# f_2 = open("/home/cw1647/phd/UCNC-experiments/results/sunspots-physical-again/size-64-n-subgroups-2-sw-0.30000000000000004-so-0.05.csv")
# f_4 = open("/home/cw1647/phd/UCNC-experiments/results/sunspots-physical-again/size-64-n-subgroups-4-sw-0.30000000000000004-so-0.025.csv")

# f2_lines = f_2.readlines()[1:]
# f4_lines = f_4.readlines()[1:]
# s = [float(line.split(",")[2]) for line in f2_lines]
# r2 = [float(line.split(",")[4]) for line in f2_lines]

# r4 = [float(line.split(",")[4]) for line in f4_lines]


# d = {"standard":s, "2 subreservoirs":r2, "4 subreservoirs":r4}
# df = pd.DataFrame(data=d)
# buf = "/home/cw1647/phd/UCNC-experiments/results/sunspots-physical-again/64.png"
# vis.ErrorVis.vis(df, buf)

# f_2 = open("/home/cw1647/phd/UCNC-experiments/results/sunspots-physical-again/size-16-n-subgroups-2-sw-0.2-so-0.025.csv")
# f_4 = open("/home/cw1647/phd/UCNC-experiments/results/sunspots-physical-again/size-16-n-subgroups-4-sw-0.2-so-0.025.csv")

# f2_lines = f_2.readlines()[1:]
# f4_lines = f_4.readlines()[1:]
# s = [float(line.split(",")[2]) for line in f2_lines]
# r2 = [float(line.split(",")[4]) for line in f2_lines]

# r4 = [float(line.split(",")[4]) for line in f4_lines]

# d = {"standard":s, "2 substates":r2, "4 substates":r4}
# df = pd.DataFrame(data=d)
# buf = "/home/cw1647/phd/UCNC-experiments/results/sunspots-physical-again/16.png"
# vis.ErrorVis.vis(df, buf)
