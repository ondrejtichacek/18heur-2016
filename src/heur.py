import numpy as np


class StopCriterion(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ShootAndGo:

    def __init__(self, of, maxeval, hmax=np.inf):
        self.of = of
        self.maxeval = maxeval
        self.fstar = self.of.get_fstar()  # local copy of obj. fun. fstar
        self.best_y = np.inf  # we will MINIMIZE the obj. fun. (!)
        self.best_x = None
        self.neval = 0
        self.hmax = hmax

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

    def report_end(self):
        return {
            'best_y': self.best_y,
            'best_x': self.best_x,
            'neval': self.neval if self.best_y <= self.fstar else np.inf
        }
