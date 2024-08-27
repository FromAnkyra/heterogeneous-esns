import NymphESN.nymphesn as nymph
import NymphESN.restrictedmatrix as rmatrix
import benchmarks.spoken_digits as digits
import numpy as np

# quick experiments to see which three channels (if any) give me the best results or if three channels is just too few

chans = [[f"{i}", f"{j}", f"{k}"] for i in range(13) for j in range(13) for k in range(13) if i<j and j<k]
# print(len(chans))
fout = open("/home/cw1647/phd/het_reservoir_experiments/spoken_digits/prelim/results/channels.txt", "a")
i = 0
print("start")
for channels in chans:
    print(i)
    data =digits.create_data(["/home/cw1647/phd/benchmarks/spoken_digits/ti46/AllTestData/", "/home/cw1647/phd/benchmarks/spoken_digits/ti46/AllTrainData"], channel_indices=channels)
    test_esn = nymph.NymphESN(3, 300, 10, density=0.1, seed=5, svd_dv=1)
    results_train, results_test = digits.run_spoken_digits(test_esn, data)
    v_digits = digits.results_to_digits(results_test, data["test coords"])
    vhat_digits = data["test digits"]
    str_out = f"{channels}, WER={digits.word_error_rate(v_digits, vhat_digits)}, matrix = {digits.confusion_matrix(v_digits, vhat_digits)}\n"
    fout.write(str_out)
    if i%3 == 0:
        fout.close()
        fout = open("/home/cw1647/phd/het_reservoir_experiments/spoken_digits/prelim/results/channels.txt", "a")
    i+=1
    # print(digits.word_error_rate(v_digits, vhat_digits))
    # print(digits.confusion_matrix(v_digits, vhat_digits))

fout.close()
