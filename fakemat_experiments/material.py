import numpy
import tempESN

class Material():
    def __init__(self, N, seed):
        self.N = N
        self.rhythm = None
        self.delay = None
        self.f = None
        self.W = self.generate_W(seed)
        self.encoding = None
        self.incoms = None
        self.outcoms = None
        return
    def generate_W(self, seed):
        pass

