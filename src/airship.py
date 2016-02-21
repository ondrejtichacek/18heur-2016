import numpy as np
import pandas as pd


class AllTroopsUsed(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class TopPeakFound(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class AirShip:

    def __init__(self):
        px = np.array([0,  50, 100, 300, 400, 700, 800], dtype=int)
        py = np.array([0, 100,   0,   0,  25,   0,  50], dtype=int)
        self.x = np.arange(0, 800+1)
        self.y = np.interp(self.x, px, py)

        self.top_peak_y = 100
        self.best_peak_x = None
        self.best_peak_y = -np.inf

        self.troops_used = 0
        self.troops_max = 100

    def eval_terrain(self, x):
        if self.troops_used < self.troops_max:
            self.troops_used += 1
            y = self.y[x]
            if y > self.best_peak_y:
                self.best_peak_y = y
                self.best_peak_x = x
            if y == self.top_peak_y:
                raise TopPeakFound('TOP_PEAK_FOUND')
            return y
        else:
            raise AllTroopsUsed('MAX_TROOPS_USED')

    def find_peak(self):
        try:
            while True:
                x = np.random.randint(800+1)
                y = self.eval_terrain(x)
                go = True
                iter_troops_used = 0
                iter_troops_max = 10
                while go and iter_troops_used < iter_troops_max:
                    y_l = -np.inf
                    y_r = -np.inf
                    if x >= 1:
                        y_l = self.eval_terrain(x-1)
                        iter_troops_used += 1
                    if x <= 799 and iter_troops_used < iter_troops_max:
                        y_r = self.eval_terrain(x+1)
                        iter_troops_used += 1
                    if np.maximum(y_l, y_r) > y:
                        go = True
                        x = x-1 if y_l > y_r else x+1
                        y = np.maximum(y_l, y_r)
                    else:
                        go = False
        except TopPeakFound:
            return {
                'best_peak_y': self.best_peak_y,
                'best_peak_x': self.best_peak_x,
                'troops_used': self.troops_used,
                'exit_reason': 'top_peak_found'
            }
        except AllTroopsUsed:
            return {
                'best_peak_y': self.best_peak_y,
                'best_peak_x': self.best_peak_x,
                'troops_used': self.troops_used,
                'exit_reason': 'all_troops_used'
            }
        except:
            print("Unexpected error")
            raise


if __name__ == "__main__":

    res = []
    for i in range(1000):
        ship = AirShip()
        res.append(ship.find_peak())

    tab = pd.DataFrame(res)
    print('Highest peak found: ' + str(tab['best_peak_y'].sort_values(ascending=False).iloc[0]))
    top_rows = tab['best_peak_y'] == 100
    print('No. of times when top peak found: ' + str(sum(top_rows)))
    print('Median of used troops when top peak found: ' + str(np.median(tab[top_rows]['troops_used'])))
    print('Median of all highest peaks found: ' + str(np.median(tab['best_peak_y'])))
