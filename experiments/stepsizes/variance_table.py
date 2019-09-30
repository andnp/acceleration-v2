import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.results import loadResults, whereParameterGreaterEq, getBest, getBestEnd
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

name = 'all'
problems = ['SmallChainTabular5050', 'SmallChainTabular4060', 'SmallChainInverted5050', 'SmallChainInverted4060', 'SmallChainDependent5050', 'SmallChainDependent4060', 'Boyan', 'Baird']

algorithms = ['gtd2', 'tdc', 'td']

if __name__ == "__main__":
    table = np.zeros((len(algorithms), len(problems), 2))

    for i, alg in enumerate(algorithms):
        for j, problem in enumerate(problems):
            exp_path = f'experiments/stepsizes/{problem}/{alg}/best_rmspbe_auc/{alg}.json'
            exp = loadExperiment(exp_path)
            results = loadResults(exp, 'variance_summary.npy')

            agent = exp.agent
            best = getBest(results)
            metric = np.mean

            if 'GTD2' in agent or 'TDC' in agent:
                best.reducer(lambda m: m[:, 1])
            else:
                best.reducer(lambda m: m[:, 1])

            mean = metric(best.mean())
            stderr = metric(best.stderr())

            table[i, j] = [mean, stderr]

    best = np.argmin(table[:, :, 0], axis=0)

    header = ' & '.join(problems)
    print('\\hline')
    print(' &' + header + '\\\\')
    print('\\hline')
    for i in range(len(algorithms)):
        elements = []
        for j in range(len(problems)):
            mean, stderr = table[i, j]
            # s = f'{mean:.3f} $\\pm$ {stderr:.3f}'
            s = f'{mean:.3f}'
            if best[j] == i:
                s = '\\textbf{' + s + '}'
            elements.append(s)

        line = ' & '.join(elements)
        line = algorithms[i] + ' & ' + line + '\\\\'
        print(line)
        print('\\hline')
