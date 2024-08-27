import matplotlib.pyplot as plt
from benchmarks.NARMA_python import NARMA10
from TinyESN import TinyESN
from TinyESN import Experiment

NARMA = NARMA10.Narma10()
esn = TinyESN.TinyESN(1, 50, 1, mode="instantaneous")
exp = Experiment.Experiment()

short_training_nrmses = []
short_testing_nrmses = []
med_training_nrmses = []
med_testing_nrmses = []
long_training_nrmses = []
long_testing_nrmses = []


NARMA.reset()
_, axs = plt.subplots(2)
esn = TinyESN.TinyESN(1, 10, 1, mode="discretised", connectivity=0.6)
short = NARMA.create_training_set(1000)
training_set = dict(list(short.items())[(len(short)-100):])
testing_set = dict(list(short.items())[:100])
esn.train_pseudoinverse(training_set)
short_training = exp.nrmse(list(training_set.values())[esn.washout_length:], esn.outputs)
axs[0].set_title("training")
axs[0].plot(list(training_set.values())[10:])
axs[0].plot(esn.outputs)
esn.test(testing_set)
short_testing = exp.nrmse(list(testing_set.values()), esn.outputs)
axs[1].set_title("testing")
axs[1].plot(list(testing_set.values()))
axs[1].plot(esn.outputs)


for i in range(50):

    NARMA.reset()
    esn = TinyESN.TinyESN(1, 10, 1, mode="discretised", connectivity=0.6)
    short = NARMA.create_training_set(500)
    training_set = dict(list(short.items())[(len(short)-100):])
    testing_set = dict(list(short.items())[:100])
    esn.train_pseudoinverse(training_set)
    short_training = exp.nrmse(list(training_set.values())[esn.washout_length:], esn.outputs)
    short_training_nrmses.append(short_training)
    esn.test(testing_set)
    short_testing = exp.nrmse(list(testing_set.values()), esn.outputs)
    short_testing_nrmses.append(short_testing)

    NARMA.reset()
    esn = TinyESN.TinyESN(1, 10, 1, mode="discretised", connectivity=0.6)
    med = NARMA.create_training_set(2000)
    training_set = dict(list(med.items())[(len(med)-100):])
    testing_set = dict(list(med.items())[:100])
    esn.train_pseudoinverse(training_set)
    med_training = exp.nrmse(list(training_set.values())[esn.washout_length:], esn.outputs)
    med_training_nrmses.append(med_training)
    esn.test(testing_set)
    med_testing = exp.nrmse(list(testing_set.values()), esn.outputs)
    med_testing_nrmses.append(med_testing)

    NARMA.reset()
    esn = TinyESN.TinyESN(1, 10, 1, mode="discretised", connectivity=0.6)
    long = NARMA.create_training_set(5000)
    training_set = dict(list(med.items())[(len(med)-100):])
    testing_set = dict(list(med.items())[:100])
    esn.train_pseudoinverse(training_set)
    long_training = exp.nrmse(list(training_set.values())[esn.washout_length:], esn.outputs)
    long_training_nrmses.append(long_training)
    esn.test(testing_set)
    long_testing = exp.nrmse(list(testing_set.values()), esn.outputs)
    long_testing_nrmses.append(long_testing)

training = [short_training_nrmses, med_training_nrmses, long_training_nrmses]
plt.boxplot(training, showfliers=False)
testing = [short_testing_nrmses, med_testing_nrmses, long_testing_nrmses]
plt.show()
