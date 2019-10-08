import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.results import loadResults, whereParameterGreaterEq, whereParameterEquals, getBest, getBestEnd
from src.utils.model import loadExperiment
from src.analysis.colormap import colors

from src.utils.arrays import first
from src.utils.path import fileName, up

error = 'rmspbe'

name = 'all'
# problems = ['SmallChainTabular5050', 'SmallChainInverted5050', 'SmallChainDependent5050', 'SmallChainTabular4060', 'SmallChainInverted4060', 'SmallChainDependent4060', 'Boyan', 'Baird']
problems = ['SmallChainTabular4060', 'SmallChainInverted4060', 'SmallChainDependent4060', 'Boyan', 'Baird']

algorithms = ['gtd2', 'tdc', 'vtrace', 'htd', 'td', 'regh_tdc']

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
    ax = plt.gca()
    f = plt.gcf()
    table = np.zeros((len(algorithms), len(problems), 2))

    for i, alg in enumerate(algorithms):
        for j, problem in enumerate(problems):
            exp_path = f'experiments/stepsizes/{problem}/{alg}/{alg}.json'
            try:
                exp = loadExperiment(exp_path)
            except:
                continue

            results = loadResults(exp, errorfile)
            if alg == 'td' or alg == 'vtrace' or alg == 'gtd2':
                const = results
            else:
                const = whereParameterEquals(results, 'ratio', 1)
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
    our_idx = indexOf(algorithms, 'regh_tdc')
    for j, problem in enumerate(problems):
        table[:, j, 0] = table[:, j, 0] / table[our_idx, j, 0]

        if htd_idx is not None and table[htd_idx, j, 0] == 0:
            table[htd_idx, j] = table[td_idx, j]

        if vtrace_idx is not None and table[vtrace_idx, j, 0] == 0:
            table[vtrace_idx, j] = table[td_idx, j]

    best = np.argmin(table[:, :, 0], axis=0)

    offset = -3
    for i, p in enumerate(problems):
        offset += 3
        for j, a in enumerate(algorithms):
            x = i * len(algorithms) + j + offset
            ax.bar(x, table[j, i, 0], yerr=table[j, i, 1], color=colors[a.upper()], tick_label='')

    ax.set_ylim([.9, 2.25])

    plt.show()
    exit()

    save_path = 'experiments/stepsizes/plots'
    os.makedirs(save_path, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/{name}_bar-plot_{error}.pdf', bbox_inches='tight', dpi=100)
