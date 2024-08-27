# train reservoirs on NARMA benchmark
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import numpy as np
# import pandas as pd
import scipy.sparse as sparse  # for sparse weight matrix
import seaborn as sns


class Reservoir:
    def __init__(self, N, rho, density, seed, bias=False):
        self.N = N  # number of nodes
        self.rho = rho      # sprectral radius
        self.density = density  # weight matric density

        np.random.seed(seed)
        self.W = self.build_random_weight_matrix()
        self.Wu = self.build_random_input_weight_vector(bias)

    def build_random_weight_matrix(self):
        # create nxn array of random numbers U[-1,1], sparse density=den
        #   normalised to max eigenvalue = 1

        w = sparse.random(self.N, self.N, density=self.density)     # sparse, uniform [0,1]
        w.data = (w.data - 0.5) * 2     # sparse, uniform [-1,1]
        w = w.toarray()     # convert to non-sparse numpy array

        # normalise w so that largest singular value is 1
        s = np.linalg.svd(w, compute_uv=False)
        w = w / s[0]
        return w

    def build_random_input_weight_vector(self, bias=False):
        # create 2xn array of random numbers U[-1,1]
        if bias:
            wu = np.random.uniform(-1, 1, size=(2, self.N))
        else:
            wu = np.random.uniform(-1, 1, size=self.N)
        return(wu)

    def set_config(self, config):

        if config == 'physical':
            self.phys_input = True
            self.phys_output = True
        else:  # config = 'flat'
            self.phys_input = False
            self.phys_output = False

    def run_reservoir(self, T, system):
        # calculate states x of reservoir, saving in system, using u in system
        # extract reservoir params
        N = self.N
        rho = self.rho
        W = self.W
        Wu = self.Wu

        # set initial x0 state to zero
        system.at[0, 'x'] = np.asarray([0] * N)

        # iterate over time to T
        for t in range(0, T - 1):
            if self.phys_input:
                u_t = system.at[t, 'u']  # <<=== use phys/prev time input
            else:
                u_t = system.at[t + 1, 'u']  # <<=== use flat/this time input

            x_t = system.at[t, 'x']
            Wu_x_u = u_t * Wu
            x_t1 = np.tanh(rho * x_t.dot(W) + Wu_x_u)  # rc eqn, x(t+1)
            system.at[t + 1, 'x'] = x_t1

    def train(self, M, D):  # M = TxN matrix of reservoir outputs; D=target column vector
        Mplus = np.linalg.pinv(M)  # pseudoinverse: M+ x M = Id[NxN] (provided T>N)
        Wv = Mplus.dot(D).T

        return Wv

    def nrmse(self, v, vhat):
        # v = list of values; vhat = list of target values
        N = len(v)
        sumsq = sum((vhat - v)**2)

        vhatmean = sum(vhat) / N
        vhatminusvhatmeansq = sum((vhat - np.asarray([vhatmean] * N))**2)

        res = math.sqrt(sumsq / vhatminusvhatmeansq)
        return res

    def train_reservoir(self, T0, T, system):

        tstart = T0         # initial t for training
        tend = T0 + T - 1   # final t for training
        M = system.iloc[tstart:tend]['x'].tolist()

        if self.phys_output:
            # output target is from next timestep
            tstart += 1
            tend += 1
        D = system.iloc[tstart:tend]['vtarget'].tolist()

        Wv = self.train(M, D)
        return Wv

    def get_reservoir_output(self, Tall, Wv, system):

        x = np.asarray(system['x'].tolist())

        v = Wv.dot(x.T).tolist()
        if self.phys_output:
            # shift forward by one timestep: v(t+1) = Wv.x(t)
            v = [0.0] + v[:-1]
        system['v'] = v

    def get_error(self, T0, Tlen, system):

        t0 = T0
        t1 = T0 + Tlen - 1
        v = system.iloc[t0:t1]['v'].to_numpy()   # convert to numpy array
        vtarget = system.iloc[t0:t1]['vtarget'].to_numpy()

        error = self.nrmse(v, vtarget)
        return(error)

# =======================================================================
# =======================================================================
# reservoir training plots


def plot_timeseries(ax, system, error, Tstart, Tlen, str):
    if Tlen > 500:
        Tlen = 500
    xs = range(Tstart, Tstart + Tlen)
    y1 = system['v'].tolist()[Tstart:Tstart + Tlen]
    y2 = system['vtarget'].tolist()[Tstart:Tstart + Tlen]
    ax.set_title(str + ' data, NMRSE = {:.3f}'.format(error),
                 y=0.98, fontsize=14)
    ax.set_xlim(Tstart, Tstart + Tlen)
    ax.set_ylim(0, 0.8)
    ax.plot(xs, y1, 'b', lw=1)
    ax.plot(xs, y2, 'darkorange', lw=1)


def plot_results(system, errordf, Tw, T1, T2, fn, boxplot_only=True):
    # Tw = washout, T1 = train, T2 = test
    # fn = filename string to save plot
    ncols = len(errordf.columns)

    sns.set_context(rc={'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18})
    fig, ax = plt.subplots(1, 1, figsize=(2.5 * ncols, 5))
    axb = ax

    my_pal = {'test ': 'gold', 'train ': 'gold'}
    sns.boxplot(data=errordf, ax=axb, notch=True, width=0.6, linewidth=0.5, fliersize=0, palette=my_pal)
    axb.set_ylabel('NRMSE')
    axb.spines['right'].set_visible(False)
    axb.spines['top'].set_visible(False)
    axb.set_ylim(0, 1.2)

    plt.show()
    #fig.savefig(fn, bbox_inches='tight')
    fig.savefig(fn)
