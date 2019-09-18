import os
import sys
import glob
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import plot, save
from src.analysis.results import loadResults, whereParameterEquals, whereParameterGreaterEq
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.path import fileName

stepsize = 'schedule'

def generatePlot(exp_paths):
    ax = plt.gca()
    # ax.semilogx()
    for exp_path in exp_paths:
        if 'lstd' in exp_path:
            continue

        if stepsize != 'constant' and stepsize not in exp_path:
            continue

        if stepsize == 'constant' and ('amsgrad' in exp_path or 'adagrad' in exp_path or 'schedule' in exp_path):
            continue

        exp = loadExperiment(exp_path)
        results = loadResults(exp, 'rmspbe_summary.npy')
        const, unconst = tee(results)

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)
        if use_ideal_h:
            continue

        const = whereParameterGreaterEq(const, 'ratio', 1)

        color = colors[exp.agent]
        label = exp.agent

        if not (exp.agent in ['TDadagrad', 'TDschedule', 'TD', 'TDamsgrad'] or use_ideal_h):
            plot(unconst, ax, label=label + '_unc', color=color, dashed=True)
            plot(const, ax, label=label, color=color, dashed=False)
        else:
            plot(unconst, ax, label=label, color=color, dashed=False)

    plt.show()
    # save(exp, f'rmsve_learning-curve')
    plt.clf()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]

    generatePlot(exp_paths)
