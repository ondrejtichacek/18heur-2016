import numpy as np


class ObjFun:
    def __init__(self, fstar, a, b):
        self.fstar = fstar
        self.a = a
        self.b = b
        pass

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
