import numpy as np
from PyExpUtils.results.paths import listResultsPaths
from src.utils.arrays import first
from src.utils.path import up

class Result:
    def __init__(self, path, exp, idx):
        self.path = path
        self.exp = exp
        self.idx = idx
        self.params = exp.getPermutation(idx)['metaParameters']
        self._data = None
        self._reducer = lambda m: m

    def _lazyLoad(self):
        if self._data is not None:
            return self._data

        try:
            self._data = np.load(self.path, allow_pickle=True)
            return self._data
        except:
            print('Result not found :: ' + self.path)
            return (np.NaN, np.NaN, 0)

    def reducer(self, lm):
        self._reducer = lm
        return self

    def mean(self):
        return self._reducer(self._lazyLoad()[0])

    def stderr(self):
        return self._reducer(self._lazyLoad()[1])

def getBestOverParameter(results, param):
    parts = {}
    for r in results:
        param_value = r.params[param]

        if param_value not in parts:
            parts[param_value] = []

        parts[param_value].append(r)

    best = {}
    for k in parts:
        best[k] = getBest(parts[k])

    return best

def getBest(results):
    low = first(results)
    for r in results:
        if np.mean(r.mean()) < np.mean(low.mean()):
            low = r

    return low

def getBestEnd(results):
    low = first(results)
    for r in results:
        a = r.mean()
        b = low.mean()
        steps = len(a) // 0.1
        if np.mean(a[-steps:]) < np.mean(b[-steps:]):
            low = r

    return low

def whereParameterEquals(results, param, value):
    return filter(lambda r: r.params[param] == value, results)

def loadResults(exp, summary='errors_summary.npy'):
    for i, path in enumerate(listResultsPaths(exp)):
        summary_path = up(path) + '/' + summary
        yield Result(summary_path, exp, i)
