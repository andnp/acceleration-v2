import os
import sys
import glob
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import plot, save, plotBest
from src.analysis.results import loadResults, whereParameterGreaterEq, whereParameterEquals, getBest, find
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

error = 'rmspbe'

# name = 'bakeoff'
# problem = 'SmallChainTabular4060'
# algorithms = ['td', 'tdc', 'htd', 'vtrace', 'regh_tdc']
# stepsize = 'constant'

name = 'broken-htd'
problem = 'Baird'
algorithms = ['tdc', 'htd', 'regh_tdc']
stepsize = 'constant'

bestBy = 'auc'
show_unconst = False
lstd_baseline = False

if error == 'rmsve':
    errorfile = 'errors_summary.npy'
elif error == 'rmspbe':
    errorfile = 'rmspbe_summary.npy'

def generatePlotTTA(ax, exp_path, bounds):
    exp = loadExperiment(exp_path)
    results = loadResults(exp, errorfile)
    const, unconst = tee(results)

    color = colors[exp.agent]
    label = exp.agent

    const = whereParameterEquals(const, 'ratio', 1)
    if 'ReghTDC' in label:
        const = whereParameterEquals(const, 'reg_h', 0.8)

    if show_unconst:
        b = plot(unconst, ax, label=label + '_unc', color=color, dashed=True, bestBy=bestBy)
        bounds.append(b)

    b = plot(const, ax, label=label, color=color, dashed=False, bestBy=bestBy)
    bounds.append(b)

def generatePlotSSA(ax, exp_path, bounds):
    exp = loadExperiment(exp_path)
    results = loadResults(exp, errorfile)

    color = colors[exp.agent]
    label = exp.agent

    b = plot(results, ax, label=label, color=color, dashed=False, bestBy=bestBy)
    bounds.append(b)


if __name__ == "__main__":
    ax = plt.gca()
    f = plt.gcf()

    bounds = []

    if lstd_baseline:
        path = f'experiments/stepsizes/{problem}/lstd.json'
        lstd_exp = loadExperiment(path)
        LSTD_res = loadResults(lstd_exp, errorfile)

        LSTD_best = getBest(LSTD_res)

        b = plotBest(LSTD_best, ax, color=colors['LSTD'], label='LSTD', alphaMain=0.5)
        bounds.append(b)

    for alg in algorithms:
        if stepsize == 'constant':
            exp_path = f'experiments/stepsizes/{problem}/{alg}/{alg}.json'
        else:
            exp_path = f'experiments/stepsizes/{problem}/{alg}/{alg}{stepsize}.json'

        if '_h' in alg or alg == 'td' or alg == 'vtrace':
            generatePlotSSA(ax, exp_path, bounds)
        else:
            generatePlotTTA(ax, exp_path, bounds)

        lower = min(map(lambda x: x[0], bounds)) * 0.9
        upper = max(map(lambda x: x[1], bounds)) * 1.05

        if lower < 0.01:
            lower = -0.01

        ax.set_ylim([lower, upper])
        ax.set_xlim([0, 150])


    # plt.show()
    # exit()

    save_path = 'experiments/stepsizes/plots'
    os.makedirs(save_path, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/{name}_learning-curve_{error}_{stepsize}_{problem}_{bestBy}.png', dpi=100)
