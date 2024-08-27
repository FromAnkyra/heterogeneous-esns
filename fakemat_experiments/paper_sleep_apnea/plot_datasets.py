import numpy as np 
import benchmarks.sleep_apnea as sleep
import matplotlib.pyplot as plt

TStart = 22000
TWashout = 1000
TTrain = 3000
TTest = 1000

TTotal = TWashout+TTrain+TTest

x = np.linspace(TStart, TStart+TTotal, TTotal)
data = sleep.get_inputs(TStart, TStart+TTotal)

fig, axen = plt.subplots(3, 1, sharex=True)

y = [-0.5, -0.5, 0.5, 0.5]

for i in range(3):
    axen[i].plot(x, data[i])
    axen[i].fill_betweenx(y, x[0], x[TWashout-1], color="green", alpha=0.3, label="Washout")
    axen[i].fill_betweenx(y, x[TWashout-1], x[TWashout+TTrain-1], color="yellow", alpha=0.3, label="Training")
    axen[i].fill_betweenx(y, x[TWashout+TTrain-1], x[-1], color="purple", alpha=0.3, label="Testing")

axen[0].set_title("heart rate", fontsize=20)
axen[1].set_title("respiration", fontsize=20)

axen[2].set_title("blood oxygenation", fontsize=20)
# fig.suptitle("Sleep Apnea Data", fontsize=24)
# axen[2].legend( loc='best', borderaxespad=0.2)
# axen[2].xtick.label.set_size(32)
axen[2].xaxis.set_tick_params(labelsize=20)

axen[0].yaxis.set_tick_params(labelsize=20)
axen[1].yaxis.set_tick_params(labelsize=20)
axen[2].yaxis.set_tick_params(labelsize=20)
plt.show()