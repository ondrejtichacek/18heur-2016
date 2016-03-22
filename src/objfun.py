import numpy as np


class ObjFun:

    def __init__(self, fstar, a, b):
        self.fstar = fstar
        self.a = a
        self.b = b

    def get_fstar(self):
        return self.fstar

    def get_bounds(self):
        return [self.a, self.b]

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


class TSPGrid(ObjFun):

    def __init__(self, par_a, par_b, norm=2):
        n = par_a * par_b  # number of cities

        # compute city coordinates
        grid = np.zeros((n, 2), dtype=np.int)
        for i in np.arange(par_a):
            for j in np.arange(par_b):
                grid[i * par_b + j]=np.array([i, j])

        # compute distances
        dist = np.zeros((n, n))
        for i in np.arange(n):
            for j in np.arange(i+1, n):
                dist[i, j] = np.linalg.norm(grid[i, :]-grid[j, :], norm)
                dist[j, i] = dist[i, j]

        self.fstar = n+np.mod(n, 2)*(2**(1/norm)-1)
        self.n = n
        self.dist = dist
        self.a = np.zeros(n-1, dtype=np.int)  # n-1 because the first city is pre-determined
        self.b = np.arange(n-2, 0-1, -1)

    def generate_point(self):
        return [np.random.randint(0, i+1) for i in np.arange(self.n-2, 0-1, -1)]

    def decode(self, x):
        #  decodes solution vector into ordered list of visited cities, e.g:
        #   x = 1 2 2 1 0
        #  cx = 2 4 5 3 1
        cx = np.zeros(self.n, dtype=np.int)  # the final tour
        ux = np.ones(self.n, dtype=np.int)  # used cities indices
        ux[0] = 0  # first city is used automatically
        c = np.cumsum(ux)  # cities to be included in the tour
        for k in np.arange(1, self.n):
            ix = x[k-1]+1  # order index of currently visited city
            cc = c[ix]  # currently visited city
            cx[k] = cc # append visited city into final tour
            c = np.delete(c, ix)  # visited city can not be included in the tour any more
        return cx

    def tour_dist(self, cx):
        d=0
        for i in np.arange(self.n):
            dx = self.dist[cx[i-1], cx[i]] if i>0 else self.dist[cx[self.n-1], cx[i]]
            d += dx
        return d

    def evaluate(self, x):
        cx = self.decode(x)
        return self.tour_dist(cx)

    def get_neighborhood(self, x, d):
        assert d == 1, "TSPGrid supports neighbourhood with distance = 1 only"
        nd = []
        for i, xi in enumerate(x):
            # x-lower
            if x[i] > self.a[i]:  # (!) mutation correction .. will be discussed later
                xl = x.copy()
                xl[i] = x[i]-1
                nd.append(xl)

            # x-upper
            if x[i] < self.b[i]:  # (!) mutation correction ..  -- // --
                xu = x.copy()
                xu[i] = x[i]+1
                nd.append(xu)

        return nd
