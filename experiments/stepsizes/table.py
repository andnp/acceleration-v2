import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.results import loadResults, whereParameterGreaterEq, whereParameterEquals, getBest, getBestEnd
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

error = 'rmspbe'

name = 'all'
problems = ['SmallChainTabular4060', 'SmallChainInverted4060', 'SmallChainDependent4060', 'Boyan', 'Baird']
stepsize = 'amsgrad'

algorithms = ['gtd2', 'tdc', 'htd', 'td', 'vtrace', 'regh_tdc']

if error == 'rmsve':
    errorfile = 'errors_summary.npy'
elif error == 'rmspbe':
    errorfile = 'rmspbe_summary.npy'


def indexOf(arr, e):
    for i, a in enumerate(arr):
        if a == e:
            return i

    return None

if __name__ == "__main__":
    table = np.zeros((len(algorithms), len(problems), 2))

    for i, alg in enumerate(algorithms):
        for j, problem in enumerate(problems):
            exp_path = f'experiments/stepsizes/{problem}/{alg}/{alg}{stepsize}.json'
            try:
                exp = loadExperiment(exp_path)
            except:
                continue

            results = loadResults(exp, errorfile)
            if alg == 'td' or alg == 'vtrace':
                const = results
            else:
                const = whereParameterGreaterEq(results, 'ratio', 1)
                const = whereParameterEquals(const, 'reg_h', 0.8)

            best = getBest(const)
            metric = np.mean

            # best = getBestEnd(const)
            # metric = lambda m: np.mean(m[-(int(len(m))):])

            mean = metric(best.mean())
            stderr = metric(best.stderr())

            table[i, j] = [mean, stderr]

    htd_idx = indexOf(algorithms, 'htd')
    vtrace_idx = indexOf(algorithms, 'vtrace')
    td_idx = indexOf(algorithms, 'td')
    for j, problem in enumerate(problems):
        if htd_idx is not None and table[htd_idx, j, 0] == 0:
            print('htd', problem)
            table[htd_idx, j] = table[td_idx, j]

        if vtrace_idx is not None and table[vtrace_idx, j, 0] == 0:
            print('vtrace', problem)
            table[vtrace_idx, j] = table[td_idx, j]

    best = np.argmin(table[:, :, 0], axis=0)

    header = ' & '.join(problems)
    print('\\hline')
    print(' &' + header + '\\\\')
    print('\\hline')
    for i in range(len(algorithms)):
        elements = []
        for j in range(len(problems)):
            mean, stderr = table[i, j]
            s = f'{mean:.3f} $\\pm$ {stderr:.3f}'
            # s = f'{mean:.3f}'
            if best[j] == i:
                s = '\\textbf{' + s + '}'
            elements.append(s)

        line = ' & '.join(elements)
        line = algorithms[i] + ' & ' + line + '\\\\'
        print(line)
        print('\\hline')
