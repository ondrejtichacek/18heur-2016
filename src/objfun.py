import numpy as np


class ObjFun:

    def __init__(self, fstar, a, b):
        self.fstar = fstar
        self.a = a
        self.b = b

    def get_fstar(self):
        return self.fstar

    def generate_point(self):
        raise NotImplementedError("Objective function must implement random point generation")

    def get_neighborhood(self, x, d):
        return x

    def evaluate(self, x):
        raise NotImplementedError("Objective function must implement objective function evaluation")


class AirShip(ObjFun):

    def generate_point(self):
        return np.random.randint(self.a, self.b+1)

    def get_neighborhood(self, x, d):
        left = [x for x in np.arange(x-1, x-d-1, -1) if x >= 0]
        right = [x for x in np.arange(x+1, x+d+1) if x <= 800]
        return np.concatenate((left, right))

    def evaluate(self, x):
        px = np.array([0,  50, 100, 300, 400, 700, 800], dtype=int)
        py = np.array([0, 100,   0,   0,  25,   0,  50], dtype=int)
        xx = np.arange(0, 800+1)
        yy = np.interp(xx, px, py)
        return -yy[x]  # negative altitude, becase we are minimizing (as opposed to the first example...)


class Zebra3(ObjFun):

    def __init__(self, d):
        self.fstar = 0
        self.d = d
        self.n = d*3
        self.a = np.zeros(self.n)
        self.b = np.ones(self.n)

    def generate_point(self):
        return np.random.randint(0, 1+1, self.n)

    def get_neighborhood(self, x, d):
        assert d == 1, "Zebra3 supports neighbourhood with (Hamming) distance = 1 only"
        nd = []
        for i, xi in enumerate(x):
            xx = x.copy()
            xx[i] = 0 if xi == 1 else 1
            nd.append(xx)
        return nd

    def evaluate(self, x):
        f = 0
        for i in np.arange(1, self.d+1):
            xr = x[(i-1)*3:i*3]
            s = np.sum(xr)
            if np.mod(i,2) == 0:
                if s == 0:
                    f += 0.9
                elif s == 1:
                    f += 0.6
                elif s == 2:
                    f += 0.3
                else:  # s == 3
                    f += 1.0
            else:
                if s == 0:
                    f += 1.0
                elif s == 1:
                    f += 0.3
                elif s == 2:
                    f += 0.6
                else:  # s == 3
                    f += 0.9
        f = self.n/3-f
        return f
