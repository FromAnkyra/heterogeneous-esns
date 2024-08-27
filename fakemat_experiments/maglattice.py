import fakemat_experiments.material as material
import numpy as np

class MagLattice(material.Material):
    def __init__(self, N, delay, seed):
        if int(np.sqrt(N)) != np.sqrt(N):
            raise ValueError("size is not a square!")
        self.svd_dv = 1
        super().__init__(N, seed)
        self.delay = delay
        self.rhythm = [1, 0, 1, 0, 1, 0] # medium reservoir
        self.encoding = "00101" #gondor
        self.incoms = np.asarray([1]*int(np.sqrt(N))+[0]*int(N-np.sqrt(N))) # 1D array of size N with the last 3/4 zeroed
        # self.outcoms = np.asarray([0]*int(N-np.sqrt(N))+[1]*int(np.sqrt(N))) #opposite "edge"
        self.outcoms = np.array(self.incoms) #same as incoms (used for multi-timescales reasons)
        return
    
    def generate_W(self, seed, debug=False):
        if debug:
            print(f"{seed=}")
        super().generate_W(seed)
        np.random.seed(seed)
        W = np.zeros((self.N, self.N))
        side = np.sqrt(self.N)
        coords_y = [[i, (i+1), (i-1), i+side, i-side] for i in range(self.N)]
        coords_y = [list(filter(lambda i: i<self.N and i>=0, l)) for l in coords_y]
        def filtersides(l):
            if l[0]%side==0:
                l.pop(2)
            if l[0]%side==side-1:
                l.pop(1)
            return l
        coords_y = [filtersides(l) for l in coords_y]
        #need to remove the "phantom" connections (ie, this is flat, not a globe)
        coords_x = [[l[0]]*len(l) for l in coords_y]

        coords_x = [x for l in coords_x for x in l]
        coords_y = [int(y) for l in coords_y for y in l]
        #flatten coords_y and coords_x

        coordinates = (coords_x, coords_y)
        # print(coordinates)
        W[coordinates] = 1
        if debug:
            density = np.sum(W)/(self.N**2)
            print(f"lattice {density=}")
            connectivity = W
        random = np.random.uniform(-0.5, 0.5, size=(self.N, self.N))
        W = W * random
        # create random weights

        s = max(abs(np.linalg.eig(W)[0]))
        svd = np.linalg.svd(W, compute_uv=False)
        if debug:
            print(f"lattice pre-norm {s=}, {svd[0]=}")
        # self.W = W
        self.W = W / (s/self.svd_dv) 
        # set svd to 1
        if debug:
            s = max(abs(np.linalg.eig(self.W)[0]))
            svd = np.linalg.svd(self.W, compute_uv=False)
            print(f"lattice (post-norm) {s=}, {svd[0]=}")
            return self.W, connectivity
        return self.W

    def rhythm_rotate(self):
        return super().rhythm_rotate()

 