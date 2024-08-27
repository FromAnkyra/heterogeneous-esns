import NymphESN.nymphesn as nymph
import NymphESN.restrictedmatrix as rmatrix
import benchmarks.spoken_digits as digits
import numpy as np

print("start data")
data =digits.create_data(["/home/cw1647/phd/benchmarks/spoken_digits/ti46/AllTestData/", "/home/cw1647/phd/benchmarks/spoken_digits/ti46/AllTrainData"])
all_digits = digits.results_to_digits(data["v train"], list(range(data["train dp"])))
print("data created")
test_esn = nymph.NymphESN(3, 300, 10, density=0.1, seed=45, svd_dv=1)
print("esn created")
results_train, results_test = digits.run_spoken_digits(test_esn, data)
print("esn has run")
v_digits = digits.results_to_digits(results_test, data["test coords"])
print("results to digits")
vhat_digits = data["test digits"]
print(digits.word_error_rate(v_digits, vhat_digits))
print(digits.confusion_matrix(v_digits, vhat_digits))