import numpy as np
import matplotlib.pyplot as plt
import NymphESN.nymphesn as nymph
import NymphESN.errorfuncs as errorfunc
import NymphESN.restrictedmatrix as rmatrix
import NymphESN.vis as vis
import benchmarks.mso as mso

def optimal_training_length(MSO, start, end, plot_name):
    i = start
    TWashout = 100 # taken from DESN paper
    TTest = 200
    x = []
    nrmse_means = []
    nrmse_sds = []
    while i <= end:
        data_lengths = (TWashout, i, TTest)
        nrmses = []
        for j in range(50):
            esn = nymph.NymphESN(1, 50, 1, seed=j)
            train, test = mso.run_MSO(esn, MSO, data_lengths)
            nrmses.append(test)
        x.append(i)
        nrmse_means.append(np.mean(nrmses))
        nrmse_sds.append(np.std(nrmses))
        i = int(i*1.05)
    #create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.plot(x, nrmse_means)
    ax2.plot(x, nrmse_sds)
    fig.savefig(plot_name)
    return    

# find optimal training lengths for different resolution for the eight sines MSO


print("resolutions")

resolution_1 = np.linspace(0, 4000, 4000)
resolution_2 = np.linspace(0, 2000, 4000)
resolution_3 = np.linspace(0, 1000, 4000)
resolution_4 = np.linspace(0, 500, 4000)

MSO_res_1 = mso.generate_MSO(resolution_1, [mso.MSO.eight.value])
MSO_res_2 = mso.generate_MSO(resolution_2, [mso.MSO.eight.value])
MSO_res_3 = mso.generate_MSO(resolution_3, [mso.MSO.eight.value])
MSO_res_4 = mso.generate_MSO(resolution_3, [mso.MSO.eight.value])

print("MSOs generated")

optimal_training_length(MSO_res_4, 200, 3000, "mso/training_lengths/training-length-resolution/train_length_res_4.png")
# optimal_training_length(MSO_res_2, 200, 3000, "mso/training_lengths/training-length-resolution/train_length_res_2.png")
# optimal_training_length(MSO_res_3, 200, 3000, "mso/training_lengths/training-length-resolution/train_length_res_3.png")

# print("training_lengths_found")
# # find optimal training lengths for different individual sines

# print("ind_sines")
# ind_sine_1 = mso.generate_MSO(resolution_1, [mso.MSO.one.value])
# ind_sine_2 = mso.generate_MSO(resolution_1, [mso.MSO.two.value])
# ind_sine_3 = mso.generate_MSO(resolution_1, [mso.MSO.three.value])
# ind_sine_4 = mso.generate_MSO(resolution_1, [mso.MSO.four.value])
# ind_sine_8 = mso.generate_MSO(resolution_1, [mso.MSO.eight.value])

# print("MSOs generated")
# optimal_training_length(ind_sine_1, 200, 3000, "mso/training_lengths/single_training_length/ind_sine_1.png")
# optimal_training_length(ind_sine_2, 200, 3000, "mso/training_lengths/single_training_length/ind_sine_2.png")
# optimal_training_length(ind_sine_3, 200, 3000, "mso/training_lengths/single_training_length/ind_sine_3.png")
# optimal_training_length(ind_sine_4, 200, 3000, "mso/training_lengths/single_training_length/ind_sine_4.png")
# optimal_training_length(ind_sine_8, 200, 3000, "mso/training_lengths/single_training_length/ind_sine_8.png")
# print("training lengths found")

# #find optimal training length for different MSOs

# print("multi_sines")

# multi_two = mso.generate_MSO(resolution_3, [mso.MSO.one.value, mso.MSO.two.value])
# multi_three = mso.generate_MSO(resolution_3, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value])
# multi_four = mso.generate_MSO(resolution_3, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value, mso.MSO.four.value, mso.MSO.four.value])
# multi_eight = mso.generate_MSO(resolution_3, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value, mso.MSO.four.value, mso.MSO.five.value, mso.MSO.six.value, mso.MSO.seven.value, mso.MSO.eight.value])

# print("MSOs generated")

# optimal_training_length(multi_two, 500, 3000, "mso/training_lengths/multi_high_res/multi_two.png")
# optimal_training_length(multi_three, 500, 3000, "mso/training_lengths/multi_high_res/multi_three.png")
# optimal_training_length(multi_four, 500, 3000, "mso/training_lengths/multi_high_res/multi_four.png")
# optimal_training_length(multi_eight, 500, 3000, "mso/training_lengths/multi_high_res/multi_eight.png")

# print("all done!")