import numpy as np
import pandas as pd
import math
import benchmarks.mso as mso
import matplotlib.pyplot as plt

def get_values(system):
    resolution_4 = np.linspace(0, 500, 4000)
    MSO_eight = mso.generate_MSO(resolution_4, [mso.MSO.one.value, mso.MSO.two.value, mso.MSO.three.value, mso.MSO.four.value, mso.MSO.five.value, mso.MSO.six.value, mso.MSO.seven.value, mso.MSO.eight.value])
    MSO_eight = MSO_eight/16
    system['vtarget'] = MSO_eight
    pass

def run_previous_output(system):
    vtarget = system['vtarget'].tolist()
    output = [0] + vtarget[:-1]
    system['v_last_result'] = output
    return

def run_average_last_two(system):
    vtarget = system['vtarget'].tolist()
    output = [0, 0] + [vtarget[i+1] + (vtarget[i+1]-vtarget[i])/2 for i in range(len(vtarget)-2)]
    system['v_last_two'] = output
    return

def get_error(col_name, system):
    v = system[col_name].to_numpy()
    vhat = system['vtarget'].to_numpy()
    N = len(v)
    sumsq = sum((vhat - v)**2)
    vhatmean = sum(vhat) / N
    vhatminusvhatmeansq = sum((vhat - np.asarray([vhatmean] * N))**2)
    res = math.sqrt(sumsq / vhatminusvhatmeansq)
    return res


system = pd.DataFrame(columns=['v_last_result', 'v_last_two', 'vtarget'])  # holds all the time states
get_values(system)
run_previous_output(system)
run_average_last_two(system)
last_result = get_error('v_last_result', system)
last_two = get_error('v_last_two', system)


print(f"{np.log(last_result)=}\n{last_two=}")