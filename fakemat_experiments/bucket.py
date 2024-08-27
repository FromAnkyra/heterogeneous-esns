import fakemat_experiments.material as material
import numpy as np

class Bucket(material.Material):
    def __init__(self, N, delay, seed):
        self.svd_dv = 0.8 # spectral radius scaling
        super().__init__(N, seed)
        self.delay = delay
        self.rhythm = [1, 0, 0, 1, 0, 0] # slowest reservoir
        # self.encoding = "00001"
        self.encoding = "01011" #gets inputs from environment and other reservoirs, does not output anything
        self.incoms = np.random.permutation([0]*(self.N-4)+[1]*4) # four random comms nodes
        self.outcoms = np.array(self.incoms) # same as incoms
        return
    
    def generate_W(self, seed, debug=False):
        np.random.seed(seed)
        if debug:
            print(f"{seed=}")
        super().generate_W(seed)
        W = np.random.uniform(-0.5, 0.5, size=(self.N, self.N))
        if debug:
            connectivity = np.ones_like(W)
        s = max(abs(np.linalg.eig(W)[0]))
        svd = np.linalg.svd(W, compute_uv=False)
        if debug:
            print(f"bucket (pre-norm) {s=}, {svd[0]=}")
            density = 1
            print(f"bucket (pre-norm) {density=}")
        # self.W = W
        self.W = W / (s/self.svd_dv) 
        # density = 1
        # low svd
        if debug:
            s = max(abs(np.linalg.eig(self.W)[0]))
            svd = np.linalg.svd(self.W, compute_uv=False)
            print(f"bucket (post-norm) {s=}, {svd[0]=}")
            density = 1
            print(f"bucket (post-norm) {density=}")
            return self.W, connectivity
        return self.W
    def rhythm_rotate(self):
        return super().rhythm_rotate()