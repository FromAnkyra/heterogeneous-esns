import numpy as np 
import benchmarks.sleep_apnea as sleep
import pandas as pd
import fakemat_experiments.create_esns as create_esns

TStart = 22000
TWashout = 1000
TTrain = 3000
TTest = 1000

data_lengths = (TStart, TWashout, TTrain, TTest)

TRuns = 100
fullsize = 64*3
subreservoir_size = 64