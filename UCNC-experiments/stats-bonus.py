from tracemalloc import Statistic
import scipy.stats as st
import statsmodels.stats.multitest as stmodel
import csv
from pathlib import Path

def analysis(small, big):
    f1 = open(small, "r")
    small_lines = f1.readlines()[1:]
    f1.close()
    f2 = open(big, "r")
    big_lines = f2.readlines()[1:]
    f2.close()
    N=50
    s_train = [float(line.split(",")[1]) for line in small_lines]
    r_train = [float(line.split(",")[3]) for line in big_lines]
    s_test = [float(line.split(",")[2]) for line in small_lines]
    r_test = [float(line.split(",")[4]) for line in big_lines]
    # print(s_train)
    # print(r_train)
    _, p_train = st.ranksums(s_train, r_train)
    _, p_test= st.ranksums(s_test, r_test)
    train_n_x_gt_y = len([i+j for i in s_train for j in r_train if i>j])
    train_n_x_e_y = len([i+j for i in s_train for j in r_train if i==j])
    R_train = N*(N+1)/2 + train_n_x_gt_y + 0.5*train_n_x_e_y
    A_train = (R_train/N - (N+1)/2)/N

    print(A_train)
    
    test_n_x_gt_y = len([i+j for i in s_test for j in r_test if i>j])
    test_n_x_e_y = len([i+j for i in s_test for j in r_test if i==j])
    R_test = N*(N+1)/2 + test_n_x_gt_y + 0.5*test_n_x_e_y
    A_test = (R_test/N - (N+1)/2)/N

    return p_train, A_train, p_test, A_test

sunspots_two = analysis("/home/cw1647/phd/UCNC-experiments/results/sunspots-physical/size-128-n-subgroups-2-sw-0.5-so-0.0.csv", "/home/cw1647/phd/UCNC-experiments/results/sunspots-physical/size-256-n-subgroups-2-sw-0.9999999999999999-so-0.0.csv")
sunspots_four = analysis("/home/cw1647/phd/UCNC-experiments/results/sunspots-physical/size-64-n-subgroups-4-sw-0.7999999999999999-so-0.1.csv", "/home/cw1647/phd/UCNC-experiments/results/sunspots-physical/size-256-n-subgroups-4-sw-0.9999999999999999-so-0.0.csv")
narma_two= analysis("/home/cw1647/phd/UCNC-experiments/results/narma-physical/size-128-subgroups-2-sw-0.1-so-0.0.csv", "/home/cw1647/phd/UCNC-experiments/results/narma-physical/size-256-subgroups-2-sw-0.1-so-0.0.csv")
narma_four = analysis("/home/cw1647/phd/UCNC-experiments/results/narma-physical/size-64-subgroups-2-sw-0.1-so-0.0.csv", "/home/cw1647/phd/UCNC-experiments/results/narma-physical/size-256-subgroups-4-sw-0.1-so-0.0.csv")

print(f"{sunspots_two=}")
print(f"{sunspots_four=}")
print(f"{narma_two=}")
print(f"{narma_four=}")

def analyse_all(results_filename, directory):
    filenames = [str(path) for path in Path(directory).glob('**/*.csv')]
    stats_full = {}
    for file in filenames:
        stats_full[file] = analysis(file)
    p_train_vals = [item[0] for item in list(stats_full.values())]
    p_test_vals = [item[2] for item in list(stats_full.values())]
    p_train_vals, p_test_vals = correct(p_train_vals, p_test_vals)
    f = open(results_filename, "w")
    writer = csv.writer(f)
    writer.writerow(["filename", "A_train", "p_train", "A_test", "p_test"])
    rows = [[filename, stats_full[filename][1], stats_full[filename][0], stats_full[filename][3], stats_full[filename][2]] for filename in list(stats_full.keys())]
    writer.writerows(rows)
    f.close()
    return

def correct(p_train_vals, p_test_vals):
    reject_test, p_test_corrected, _, alpha = stmodel.multipletests(p_test_vals, method='b')
    reject_train, p_train_corrected, _, alpha = stmodel.multipletests(p_train_vals, method='b')
    return p_train_corrected, p_test_corrected
