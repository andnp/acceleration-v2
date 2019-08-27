import numpy as np
import sys
import json
from src.utils.model import loadExperiment
from PyExpUtils.results.paths import listResultsPaths

def loadPaths(p):
    p = '/'.join(p.split('/')[:-1])
    try:
        a = np.load(f'{p}/rmsve_summary.npy')
        return (p, a[0], a[1], [2])
    except:
        print('failed to load', p)
        return (p, np.inf, 0, 1 )

for exp_path in sys.argv[1:]:
    exp = loadExperiment(exp_path)
    paths = listResultsPaths(exp, runs=1)

    results = map(loadPaths, paths)

    best = 1000
    params = None
    for res in results:
        path, m, stderr, count = res
        e = np.mean(m)

        # print(path, e)

        if e < best:
            best = e
            params = path

    print(best, params)
