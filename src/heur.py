import numpy as np


class StopCriterion(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Heuristic:

    def __init__(self, of, maxeval):
        self.of = of
        self.maxeval = maxeval
        self.fstar = of.get_fstar()  # local copy of obj. fun. fstar
        [self.a, self.b] = of.get_bounds()  # local copy of obj. fun. domain bounds
        self.best_y = np.inf
        self.best_x = None
        self.neval = 0
        self.step_data = None

    def evaluate(self, x):
        y = self.of.evaluate(x)
        self.neval += 1
        if y < self.best_y:
            self.best_y = y
            self.best_x = np.copy(x)
        if y <= self.fstar:
            raise StopCriterion('Found solution with desired fstar value')
        if self.neval == self.maxeval:
            raise StopCriterion('Exhausted maximum allowed number of evaluations')
        return y

    def append_log(self, step, params):
        for param in params:
            self.step_data[param][step] = params[param]

    def report_end(self):
        return {
            'best_y': self.best_y,
            'best_x': self.best_x,
            'neval': self.neval if self.best_y <= self.fstar else np.inf,
            'step_data': self.step_data
        }


class ShootAndGo(Heuristic):

    def __init__(self, of, maxeval, hmax=np.inf):
        Heuristic.__init__(self, of, maxeval)
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
        Heuristic.__init__(self, of, maxeval)

        self.T0 = T0
        self.n0 = n0
        self.alpha = alpha
        self.r = r
        self.step_data = {
            'T': np.empty(maxeval)*np.nan,
            'mut_size': np.empty(maxeval)*np.nan,
            'x': np.zeros(maxeval, dtype=int)*np.nan,
            'f_x': np.empty(maxeval)*np.nan,
            'y': np.zeros(maxeval, dtype=int)*np.nan,
            'f_y': np.empty(maxeval)*np.nan
        }

    def mutate(self, x):
        # Discrete Cauchy mutation (TO BE GENERALIZED!)
        n = np.size(x)
        u = np.random.uniform(low=0.0, high=1.0, size=n)
        r = self.r
        x_new = x + r*np.tan(np.pi * (u-1/2))

        x_new_corrected = np.minimum(np.maximum(x_new, self.a), self.b)
        return np.array(np.round(x_new_corrected), dtype=int)

    def search(self):
        try:
            x = self.of.generate_point()
            f_x = self.evaluate(x)
            Heuristic.append_log(self, 0, {'x': x, 'f_x': f_x})
            while True:
                k = self.neval
                T0 = self.T0
                n0 = self.n0
                alpha = self.alpha

                y = self.mutate(x)
                Heuristic.append_log(self, k, {'x': x, 'f_x': f_x, 'mut_size': np.linalg.norm(x-y), 'y': y})
                f_y = self.evaluate(y)

                T = T0/(1+(k/n0)**alpha) if alpha > 0 else T0*np.exp(-(k/n0)**-alpha)
                s = (f_x - f_y)/T
                if np.random.uniform() < 1/2 + np.arctan(s)/np.pi:
                    x = y
                    f_x = f_y
                Heuristic.append_log(self, k, {'T': T})

        except StopCriterion:
            return self.report_end()
        except:
            raise


class GO(Heuristic):

    def __init__(self, of, maxeval, n, m, t_sel1, t_sel2, r, co_m):
        Heuristic.__init__(self, of, maxeval)

        assert m > n, 'M should be larger than N'
        self.n = n  # population size
        self.m = m  # working population size
        self.t_sel1 = t_sel1  # first selection temperature
        self.t_sel2 = t_sel2  # second selection temperature
        self.r = r  # mutation radius
        self.co_m = co_m  # number of crossover points    m=m+1  # m = number of crossover points

    @staticmethod
    def sort_pop(pop_x, pop_f):
        ixs = np.argsort(pop_f)
        pop_x = pop_x[ixs]
        pop_f = pop_f[ixs]
        return [pop_x, pop_f]

    @staticmethod
    def rank_select(temp, n_max):
        u = np.random.uniform(low=0.0, high=1.0, size=1)
        ix = np.minimum(np.ceil(-temp*np.log(u)), n_max)-1
        return ix.astype(int)

    def mutate(self, x):
        # Discrete Cauchy mutation (TO BE GENERALIZED!)
        n = np.size(x)
        u = np.random.uniform(low=0.0, high=1.0, size=n)
        r = self.r
        x_new = x + r*np.tan(np.pi * (u-1/2))

        x_new_corrected = np.minimum(np.maximum(x_new, self.a), self.b)
        return np.array(np.round(x_new_corrected), dtype=int)

    def crossover(self, x, y):
        m = self.co_m+1  # m = number of crossover points
        n = np.size(x)
        z = x*0
        k = 0
        p = np.ceil(n/m).astype(int)
        for i in np.arange(1, m+1):
            ix_from = k
            ix_to = np.minimum(k+p, n)
            z[ix_from:ix_to] = x[ix_from:ix_to] if np.mod(i, 2) == 1 else y[ix_from:ix_to]
            k += p
        return z

    def search(self):
        try:
            # Initialization:
            pop_x = np.zeros([self.n, np.size(self.a)], dtype=int)  # population solution vectors
            pop_f = np.zeros(self.n)  # population fitness (objective) function values
            # a.) generate the population
            for i in np.arange(self.n):
                x = self.of.generate_point()
                pop_x[i, :] = x
                pop_f[i] = self.evaluate(x)

            # b.) sort according to fitness function
            [pop_x, pop_f] = self.sort_pop(pop_x, pop_f)

            # Evolution iteration
            while True:
                # 1.) generate the working population
                work_pop_x = np.zeros([self.m, np.size(self.a)], dtype=int)
                work_pop_f = np.zeros(self.m)
                for i in np.arange(self.m):
                    par_a_ix = self.rank_select(temp=self.t_sel1, n_max=self.n)  # select first parent
                    par_b_ix = self.rank_select(temp=self.t_sel1, n_max=self.n)  # select second parent (uniqueness not guaranteed!)
                    par_a = pop_x[par_a_ix, :][0]
                    par_b = pop_x[par_b_ix, :][0]
                    z = self.crossover(par_a, par_b)
                    z_mut = self.mutate(z)
                    work_pop_x[i, :] = z_mut
                    work_pop_f[i] = self.evaluate(z_mut)

                # 2.) sort working population according to fitness function
                [work_pop_x, work_pop_f] = self.sort_pop(work_pop_x, work_pop_f)

                # 3.) select the new population
                ixs_not_selected = np.ones(self.m, dtype=bool)  # this mask will prevent us from selecting duplicates
                for i in np.arange(self.n):
                    sel_ix = self.rank_select(temp=self.t_sel2, n_max=np.sum(ixs_not_selected))
                    pop_x[i, :] = work_pop_x[ixs_not_selected][sel_ix, :]
                    pop_f[i] = work_pop_f[ixs_not_selected][sel_ix]
                    ixs_not_selected[sel_ix] = False

                # 4.) sort according to fitness function
                [pop_x, pop_f] = self.sort_pop(pop_x, pop_f)

        except StopCriterion:
            return self.report_end()
        except:
            raise
