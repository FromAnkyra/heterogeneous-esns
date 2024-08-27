from benchmarks.mso import *

def single_density(density, N):
    return

def optimal_dw(MSO):
    density = 0.01
    densities = dict()
    while density < 1:
        nrmse = single_density(density)

