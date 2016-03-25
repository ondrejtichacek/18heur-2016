import numpy as np


class StopCriterion(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Heuristic:

    def __init__(self, of, maxeval):
        raise NotImplementedError("Heuristic must implement an initialization")

    def evaluate(self, x):
        y = self.of.evaluate(x)
        self.neval += 1
        if y < self.best_y:
            self.best_y = y
            self.best_x = x
        if y <= self.fstar:
            raise StopCriterion('Found solution with desired fstar value')
        if self.neval == self.maxeval:
            raise StopCriterion('Exhausted maximum allowed number of evaluations')
        return y

    def report_end(self):
        return {
            'best_y': self.best_y,
            'best_x': self.best_x,
            'neval': self.neval if self.best_y <= self.fstar else np.inf
        }


class ShootAndGo(Heuristic):

    def __init__(self, of, maxeval, hmax=np.inf):
        self.of = of
        self.maxeval = maxeval
        self.fstar = self.of.get_fstar()  # local copy of obj. fun. fstar
        self.best_y = np.inf  # we will MINIMIZE the obj. fun. (!)
        self.best_x = None
        self.neval = 0
        self.hmax = hmax

    def steepest_descent(self, x):
        # Steepest (Hill) Descent beginning in x
        desc_best_y = np.inf
        desc_best_x = x
        h = 0
        go = True
        while go and h < self.hmax:
            go = False
            nhood = self.of.get_neighborhood(desc_best_x, 1)
            for xn in nhood:
                yn = self.evaluate(xn)
                h += 1
                if yn < desc_best_y:
                    desc_best_y = yn
                    desc_best_x = xn
                    go = True
                if h == self.hmax:
                    go = False

    def search(self):
        try:
            while True:
                # Random Shoot...
                x = self.of.generate_point()  # global search
                self.evaluate(x)
                # ...and Go (optional)
                if self.hmax > 0:
                    self.steepest_descent(x)  # local search

        except StopCriterion:
            return self.report_end()
        except:
            raise


class FSA(Heuristic):

    def __init__(self, of, maxeval, T0, n0, alpha, r):
        self.of = of
        self.maxeval = maxeval
        self.fstar = self.of.get_fstar()  # local copy of obj. fun. fstar
        [self.a, self.b] = self.of.get_bounds()  # local copy of obj. fun. domain bounds
        self.best_y = np.inf
        self.best_x = None
        self.neval = 0
        self.T0 = T0
        self.n0 = n0
        self.alpha = alpha
        self.r = r

    def mutate(self, x):
        # Discrete Cauchy mutation
        n = np.size(x)
        u = np.random.uniform(low=0.0, high=1.0, size=n)
        r = self.r
        x_new = x + r*np.tan(np.pi * (u-1/2))

        a = self.a
        b = self.b
        x_new_corrected = np.minimum(np.maximum(x_new, self.a), self.b)
        return np.round(x_new_corrected)

    def search(self):
        try:
            x = self.of.generate_point()
            f_x = self.evaluate(x)
            while True:
                k = self.neval
                T0 = self.T0
                n0 = self.n0
                alpha = self.alpha

                y = self.mutate(x)
                f_y = self.evaluate(y)

                T = T0/(1+(k/n0)**alpha) if alpha > 0 else T0*np.exp(-(k/n0)**-alpha)
                s = (f_x - f_y)/T
                if np.random.uniform() < 1/2 + np.arctan(s)/np.pi:
                    x = y
                    f_x = f_y

        except StopCriterion:
            return self.report_end()
        except:
            raise

