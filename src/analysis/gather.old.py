import numpy as np
import sys
import os
from multiprocessing import Pool
from glob import glob
from src.utils.model import loadExperiment
from PyExpUtils.results.paths import listResultsPaths

def loadPaths(p):
    p = '/'.join(p.split('/')[:-1])

    m = []
    for path in glob(f'{p}/*/errors.npy'):
        try:
            a = np.load(path)
            m.append(a)
        except:
            print('failed to load result', path)

    mean = np.mean(np.array(m), axis=0)
    stdev = np.std(np.array(m), axis=0)

    length = len(m)

    return (p, mean, stdev / np.sqrt(length), length)

def save_summary(new_base, exp_path, pool):
    exp = loadExperiment(exp_path)
    paths = listResultsPaths(exp, runs=1)

    results = pool.map(loadPaths, paths)

    for res in results:
        path, m, stderr, count = res

        main_path = '/'.join(path.split('/')[1:])
        new_path = f'{new_base}/{main_path}'
        os.makedirs(new_path, exist_ok=True)
        np.save(f'{new_path}/errors_summary.npy', np.array([m, stderr, count]))

        print(main_path, count, np.mean(m), np.mean(stderr))


if __name__ == '__main__':
    new_base = sys.argv[1]
    if new_base.endswith('.json'):
        raise Exception('uh-oh. looks like you forgot to specify the new base as the first arg')

    exps = sys.argv[2:]
    pool = Pool(32)

    for exp in exps:
        save_summary(new_base, exp, pool)
