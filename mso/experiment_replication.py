import numpy as np 
import benchmarks.mso as mso
import NymphESN.nymphesn as nymph
import NymphESN.restrictedmatrix as rmatrix
import matplotlib.pyplot as plt


# run MSO 2 on 50 reservoirs (use sparsity as decided in density.py?)
# boxplot the error both MSE and NRMSE
# maybe do it for 3, 4, and 8 too? to get a good idea of what to compare against

resolution_3 = np.linspace(0, 1000, 1000)
DW = 0.05
DB = 0.0125
N = 400

TWashout = 100
TTrain = 600
TTest = 300

# standard reservoir, MSO 2
# restricted reservoir, MSO 2
print("starting multi-2")
# multi_two = mso.generate_MSO(resolution_3, [mso.MSO.one.value, mso.MSO.two.value])
# s_nrmse = []
# r_nrmse = []
# s_mse = []
# r_mse = []
# for i in range(3000):
#     standard = nymph.NymphESN(1, N, 1, seed=i, density=0.05)
#     restricted = nymph.NymphESN(1, N, 1, seed=i)
#     W = rmatrix.create_restricted_esn_weights(N, 100, 4, within_connectivity=DW, outwith_connectivity=DB)
#     standard.set_data_lengths(TWashout, TTrain, TTest)
#     restricted.set_data_lengths(TWashout, TTrain, TTest)
#     ((s_nr_train, s_nr_test),(s_test, s_train)) = mso.run_MSO(standard, multi_two, (TWashout, TTrain, TTest), error="both")
#     ((r_nr_train, r_nr_test),(r_test, r_train)) = mso.run_MSO(standard, multi_two, (TWashout, TTrain, TTest), error="both")
#     s_nrmse.append(s_nr_test)
#     r_nrmse.append(r_nr_test)
#     s_mse.append(s_train)
#     r_mse.append(r_train)

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.boxplot([s_nrmse, r_nrmse], labels = ["standard", "restricted"])
# ax1.set_title("nrmses")

# ax2.boxplot([s_mse, r_mse], labels=["standard", "restricted"])
# ax2.set_title("mses")

# fig.savefig("mso/error_measure/multi_two.png")
# standard reservoir, MSO 3
# restricted reservoir, MSO 3
print("starting multi-3")
# multi_three = mso.generate_MSO(resolution_3, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value])

# s_nrmse = []
# r_nrmse = []
# s_mse = []
# r_mse = []
# for i in range(3000):
#     standard = nymph.NymphESN(1, N, 1, seed=i, density=0.05)
#     restricted = nymph.NymphESN(1, N, 1, seed=i)
#     W = rmatrix.create_restricted_esn_weights(N, 100, 4, within_connectivity=DW, outwith_connectivity=DB)
#     standard.set_data_lengths(TWashout, TTrain, TTest)
#     restricted.set_data_lengths(TWashout, TTrain, TTest)
#     ((s_nr_train, s_nr_test),(s_test, s_train)) = mso.run_MSO(standard, multi_three, (TWashout, TTrain, TTest), error="both")
#     ((r_nr_train, r_nr_test),(r_test, r_train)) = mso.run_MSO(standard, multi_three, (TWashout, TTrain, TTest), error="both")
#     s_nrmse.append(s_nr_test)
#     r_nrmse.append(r_nr_test)
#     s_mse.append(s_train)
#     r_mse.append(r_train)

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.boxplot([s_nrmse, r_nrmse], labels = ["standard", "restricted"])
# ax1.set_title("nrmses")

# ax2.boxplot([s_mse, r_mse], labels=["standard", "restricted"])
# ax2.set_title("mses")

# fig.savefig("mso/error_measure/multi_three.png")

# standard reservoir, MSO 4
# restricted reservoir, MSO 4
print("starting multi-4")
multi_four = mso.generate_MSO(resolution_3, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value, mso.MSO.four.value, mso.MSO.four.value])

s_nrmse = []
r_nrmse = []
s_mse = []
r_mse = []
for i in range(3000):
    standard = nymph.NymphESN(1, N, 1, seed=i, density=0.05)
    restricted = nymph.NymphESN(1, N, 1, seed=i)
    W = rmatrix.create_restricted_esn_weights(N, 100, 4, within_connectivity=DW, outwith_connectivity=DB)
    standard.set_data_lengths(TWashout, TTrain, TTest)
    restricted.set_data_lengths(TWashout, TTrain, TTest)
    ((s_nr_train, s_nr_test),(s_test, s_train)) = mso.run_MSO(standard, multi_four, (TWashout, TTrain, TTest), error="both")
    ((r_nr_train, r_nr_test),(r_test, r_train)) = mso.run_MSO(standard, multi_four, (TWashout, TTrain, TTest), error="both")
    s_nrmse.append(s_nr_test)
    r_nrmse.append(r_nr_test)
    s_mse.append(s_train)
    r_mse.append(r_train)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.boxplot([s_nrmse, r_nrmse], labels = ["standard", "restricted"])
ax1.set_title("nrmses")

ax2.boxplot([s_mse, r_mse], labels=["standard", "restricted"])
ax2.set_title("mses")

fig.savefig("mso/error_measure/multi_four.png")

# standard reservoir, MSO 8
# restricted reservoir, MSO 8
print("starting multi-8")
multi_eight = mso.generate_MSO(resolution_3, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value, mso.MSO.four.value, mso.MSO.five.value, mso.MSO.six.value, mso.MSO.seven.value, mso.MSO.eight.value])

s_nrmse = []
r_nrmse = []
s_mse = []
r_mse = []
for i in range(3000):
    standard = nymph.NymphESN(1, N, 1, seed=i, density=0.05)
    restricted = nymph.NymphESN(1, N, 1, seed=i)
    W = rmatrix.create_restricted_esn_weights(N, 100, 4, within_connectivity=DW, outwith_connectivity=DB)
    standard.set_data_lengths(TWashout, TTrain, TTest)
    restricted.set_data_lengths(TWashout, TTrain, TTest)
    ((s_nr_train, s_nr_test),(s_test, s_train)) = mso.run_MSO(standard, multi_eight, (TWashout, TTrain, TTest), error="both")
    ((r_nr_train, r_nr_test),(r_test, r_train)) = mso.run_MSO(standard, multi_eight, (TWashout, TTrain, TTest), error="both")
    s_nrmse.append(s_nr_test)
    r_nrmse.append(r_nr_test)
    s_mse.append(s_train)
    r_mse.append(r_train)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.boxplot([s_nrmse, r_nrmse], labels = ["standard", "restricted"])
ax1.set_title("nrmses")

ax2.boxplot([s_mse, r_mse], labels=["standard", "restricted"])
ax2.set_title("mses")

fig.savefig("mso/error_measure/multi_eight.png")