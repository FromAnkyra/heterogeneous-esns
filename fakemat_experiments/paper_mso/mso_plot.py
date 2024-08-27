import numpy as np
import benchmarks.mso as mso
import matplotlib.pyplot as plt

x = np.linspace(0, 500, 4000)
x_prime = np.linspace(0, 4000, 4000)
x_sample = x[0:240] 
x_prime_sample = x_prime[0:240]
MSO_eight = mso.generate_MSO(x, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value, mso.MSO.four.value, mso.MSO.five.value, mso.MSO.six.value, mso.MSO.seven.value, mso.MSO.eight.value])/16
MSO_four = mso.generate_MSO(x, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value, mso.MSO.four.value])/8
MSO_two = mso.generate_MSO(x, [mso.MSO.one.value, mso.MSO.two.value])/4


trad_fast_x = [x_sample[i] for i in range(len(x_sample)) if i%8==0]
trad_med_x = [trad_fast_x[i] for i in range(len(trad_fast_x)) if i%2==0]
trad_slow_x = [trad_fast_x[i] for i in range(len(trad_fast_x)) if i%3==0]

trad_fast_y = [MSO_eight[i] for i in range(len(x_sample)) if i%8==0]
trad_med_y = [trad_fast_y[i] for i in range(len(trad_fast_y)) if i%2==0]
trad_slow_y = [trad_fast_y[i] for i in range(len(trad_fast_y)) if i%3==0]

ours_fast_x = [x_prime[i] for i in range(len(x_sample))]
ours_med_x = [ours_fast_x[i] for i in range(len(ours_fast_x)) if i%2==0]
ours_slow_x = [ours_fast_x[i] for i in range(len(ours_fast_x)) if i%3==0]

ours_fast_y = [MSO_eight[i] for i in range(len(x_sample))]
ours_med_y = [ours_fast_y[i] for i in range(len(ours_fast_y)) if i%2==0]
ours_slow_y = [ours_fast_y[i] for i in range(len(ours_fast_y)) if i%3==0]

sample_fig, sample_ax = plt.subplots(2, 3, sharex=False, sharey=True)

for j in range(3):
    sample_ax[0, j].plot(x_sample, MSO_eight[0:len(x_sample)])
    sample_ax[1, j].plot(x_prime_sample, MSO_eight[0:len(x_sample)])

sample_ax[0, 0].plot(trad_fast_x, trad_fast_y, 'ro')
sample_ax[0, 1].plot(trad_med_x, trad_med_y, 'ro')
sample_ax[0, 2].plot(trad_slow_x, trad_slow_y, 'ro')

sample_ax[1, 0].plot(ours_fast_x, ours_fast_y, 'r.')
sample_ax[1, 1].plot(ours_med_x, ours_med_y, 'r.')
sample_ax[1, 2].plot(ours_slow_x, ours_slow_y, 'r.')

sample_fig.suptitle("Datapoints Seen by Each Reservoir Using Different Sampling Techniques", size='xx-large')

cols = ["fastest reservoir", "medium reservoir", "slowest reservoir"]
for c, ax in zip(cols, sample_ax[0]):
    ax.set_title(c, size='x-large')

rows = ["traditional\nsampling", "our\nsampling"]
for r, ax in zip(rows, sample_ax[:, 0]):
    ax2 = ax.twinx()
    # move extra axis to the left, with offset
    ax2.yaxis.set_label_position('left')
    ax2.spines['left'].set_position(('axes', -0.1))
    # hide spine and ticks, set group label
    ax2.spines['left'].set_visible(False)
    ax2.set_yticks([])
    ax2.set_ylabel(r, rotation=0, size='x-large',
                   ha='right', va='center')

x_datlen = x_prime[0:1100]
mso8_datlen = MSO_eight[0:1100]
mso4_datlen = MSO_four[0:1100]
mso2_datlen = MSO_two[0:1100]

datlenfig, datlenax = plt.subplots(1, 3, sharex=True)

datlenax[0].plot(x_datlen, mso2_datlen)
datlenax[1].plot(x_datlen, mso4_datlen)
datlenax[2].plot(x_datlen, mso8_datlen)

datlenax[0].set_title("MSO*-2", size='x-large')
datlenax[1].set_title("MSO*-4", size='x-large')
datlenax[2].set_title("MSO*-8", size='x-large')
# datlenfig.suptitle("Data used for the MSO tasks", size='xx-large')

x0 = x_datlen[0]
xw = x_datlen[99]
xtr = x_datlen[899]
xte = x_datlen[1099]

y0 = [-0.5, -0.5, 0.5, 0.5]
y1 = [-4, -4, 4, 4]
y2 = [-8, -8, 8, 8]

datlenax[0].fill_betweenx(y0, x0, xw, color="green", alpha=0.3, label="Washout")
datlenax[0].fill_betweenx(y0, xw, xtr, color="yellow", alpha=0.3, label="Training")
datlenax[0].fill_betweenx(y0, xtr, xte, color="purple", alpha=0.3, label="Testing")

datlenax[1].fill_betweenx(y0, x0, xw, color="green", alpha=0.3)
datlenax[1].fill_betweenx(y0, xw, xtr, color="yellow", alpha=0.3)
datlenax[1].fill_betweenx(y0, xtr, xte, color="purple", alpha=0.3)

datlenax[2].fill_betweenx(y0, x0, xw, color="green", alpha=0.3, label="Washout")
datlenax[2].fill_betweenx(y0, xw, xtr, color="yellow", alpha=0.3, label="Training")
datlenax[2].fill_betweenx(y0, xtr, xte, color="purple", alpha=0.3, label="Testing")

datlenax[2].legend( loc='best', borderaxespad=0.2)

plt.show()