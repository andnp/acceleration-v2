import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.sensitivity_curve import plotSensitivity, save
from src.analysis.results import loadResults, whereParameterGreaterEq, whereParameterEquals, getBest, getBestEnd, find
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

error = 'rmspbe'

name = 'bakeoff'
problem = 'SmallChainInverted4060'
algorithms = ['tdc', 'td', 'regh_tdc']
# algorithms = ['tdc', 'td', 'regh_tdc']
stepsize = 'constant'
param = 'alpha'

# name = 'broken-htd'
# problem = 'Baird'
# algorithms = ['tdc', 'htd', 'regh_tdc']
# stepsize = 'constant'

bestBy = 'auc'
show_unconst = False
td_baseline = False

SMALL = 8
MEDIUM = 16
BIGGER = 20

plt.rc('font', size=SMALL)          # controls default text sizes
plt.rc('axes', titlesize=SMALL)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM)    # legend fontsize
plt.rc('figure', titlesize=BIGGER)  # fontsize of the figure title

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

    const = whereParameterEquals(const, 'ratio', 1.0)

    if 'ReghTDC' in label:
        const = whereParameterEquals(const, 'reg_h', 0.8)

    if show_unconst:
        b = plotSensitivity(unconst, param, ax, color=color, label=label + '_unc', bestBy=bestBy, dashed=True)
        bounds.append(b)

    b = plotSensitivity(const, param, ax, color=color, label=label, bestBy=bestBy)
    bounds.append(b)

def generatePlot(ax, exp_path, bounds):
    exp = loadExperiment(exp_path)
    results = loadResults(exp, errorfile)

    color = colors[exp.agent]
    label = exp.agent

    b = plotSensitivity(results, param, ax, color=color, label=label, bestBy=bestBy)
    bounds.append(b)

if __name__ == "__main__":
    ax = plt.gca()
    f = plt.gcf()

    bounds = []

    if td_baseline:
        if stepsize == 'constant':
            path = f'experiments/stepsizes/{problem}/td/td.json'
        else:
            path = f'experiments/stepsizes/{problem}/td/td{stepsize}.json'

        td_exp = loadExperiment(path)
        TD_res = loadResults(td_exp, errorfile)

        if bestBy == 'end':
            metric = lambda m: np.mean(m[-int(m.shape[0] * .1):])
            best = getBestEnd(TD_res)
        elif bestBy == 'auc':
            metric = np.mean
            best = getBest(TD_res)

        m = metric(best.mean())
        ax.hlines(m, 2**-6, 2**6, color=colors['TD'], label='TD', linewidth=2)

    for alg in algorithms:
        if stepsize == 'constant':
            exp_path = f'experiments/stepsizes/{problem}/{alg}/{alg}.json'
        else:
            exp_path = f'experiments/stepsizes/{problem}/{alg}/{alg}{stepsize}.json'

        if alg != 'td':
            generatePlotTTA(ax, exp_path, bounds)
        else:
            generatePlot(ax, exp_path, bounds)

    lower = min(map(lambda x: x[0], bounds)) * 0.9
    upper = max(map(lambda x: x[1], bounds)) * 1.05

    if lower < 0.01:
        lower = -0.01

    ax.set_ylim([lower, upper])
    ax.set_xscale("log", basex=2)

    plt.show()
    exit()

    save_path = 'experiments/stepsizes/plots'
    os.makedirs(save_path, exist_ok=True)

    # merp. backwards compatibility with old file names
    if param == 'ratio':
        param = 'eta'

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/{name}_{param}-sensitivity_{error}_{stepsize}_{problem}_{bestBy}.pdf', bbox_inches='tight', dpi=100)
