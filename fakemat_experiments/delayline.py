import numpy as np
import scipy.sparse as sparse
import fakemat_experiments.material as material

class DelayLine(material.Material):
    def __init__(self, N, delay, seed):
        self.svd_dv = 1
        super().__init__(N, seed)
        self.delay = delay
        self.rhythm = [1, 1, 1, 1, 1, 1] # fastest reservoir
        self.encoding = "00001" # no communication when asleep
        # self.encoding = "01011" #gets inputs from environment and other reservoirs, does not output anything
        self.incoms = np.random.permutation([0]*(self.N-1)+[1]) # one
        self.outcoms = np.random.permutation([0]*(self.N-1)+[1]) #one
        return
    
    def generate_W(self, seed, debug=False):
        if debug:
            print(f"{seed=}")
        super().generate_W(seed)
        np.random.seed(seed)
        W = np.zeros((self.N, self.N))
        coords_x = [x for l in [[i, i] for i in range(self.N)] for x in l]
        coords_y = [y for l in [[i, (i+1)%self.N] for i in range(self.N)] for y in l]

        coordinates = (coords_x, coords_y)
        W[coordinates] = 1
        if debug:
            density = np.sum(W)/(self.N**2)
            print(f"ring {density=}")
            connectivity = W
        # ring topology with recurrent connections

        random = np.random.uniform(-0.5, 0.5, size=(self.N, self.N))
        W = W * random
        # create random weights

        s = max(abs(np.linalg.eig(W)[0]))
        svd = np.linalg.svd(W, compute_uv=False)
        if debug:
            print(f"ring pre-norm {s=}, {svd[0]=}")
        # self.W = W
        self.W = W / (s/self.svd_dv) 
        # set svd to 1
        if debug:
            s = max(abs(np.linalg.eig(self.W)[0]))
            svd = np.linalg.svd(self.W, compute_uv=False)
            print(f"ring (post-norm) {s=}, {svd[0]=}")
            return self.W, connectivity
        return W
    def rhythm_rotate(self):
        return super().rhythm_rotate()