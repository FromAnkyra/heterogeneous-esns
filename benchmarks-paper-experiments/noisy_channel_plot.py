import numpy as np 
import matplotlib.pyplot as plt

def make_u(N):
    choices = np.asarray([-3, -1, 1, 3])
    return np.random.choice(choices, (100,))

def noiseless(u):
    q = []
    for i in range(7, len(u)-2):
        q_n = 0.08 * u[i+2] - 0.12 * u[i+1] + u[i] + 0.18 * u[i-1] - 0.1 * u[i-2] + 0.09 * u[i-1] - 0.05 * u[i-4] + 0.04 * u[i-5] + 0.03 * u[i-6] + 0.01 * u[i-7]
        q.append(q_n)
    return q

def noisy(q, v):
    return map(lambda x, y : x + (0.036*x)**2 - (0.011*x)**3 + y, q, v)

u = make_u(110)
q = noiseless(u)
v = np.random.normal(loc=22.0, scale=4.0, size=110)
d = noisy(q, v)
v_none = [0]*110
no_v = noisy(q, v_none)
plt.plot(list(q), label="original")
plt.plot(list(d), label="noisy with offset")
plt.legend()
plt.show()

    

