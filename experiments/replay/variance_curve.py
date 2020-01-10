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

name = 'bakeoff'
problem = 'SmallChainInverted4060'
algorithms = ['tdc', 'gtd2', 'htd', 'regh_tdc']
stepsize = 'constant'

# name = 'broken-htd'
# problem = 'Baird'
# algorithms = ['tdc', 'htd', 'regh_tdc']
# stepsize = 'constant'

window = 1
smoothing = 0.1

SMALL = 8
MEDIUM = 11
BIGGER = 14

plt.rc('font', size=SMALL)          # controls default text sizes
plt.rc('axes', titlesize=SMALL)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER)    # legend fontsize
plt.rc('figure', titlesize=BIGGER)  # fontsize of the figure title

def generatePlot(ax, exp_path, bounds):
    exp = loadExperiment(exp_path)
    results = loadResults(exp, 'variance_summary.npy')

    best = getBest(results)
    best.reducer(lambda m: m[:, 0])

    color = colors[exp.agent]
    label = exp.agent

    b = plotBest(best, ax, window=window, smoothing=smoothing, label=label, color=color, dashed=False)
    bounds.append(b)


if __name__ == "__main__":
    ax = plt.gca()
    f = plt.gcf()

    bounds = []

    for alg in algorithms:
        if stepsize == 'constant':
            exp_path = f'experiments/stepsizes/{problem}/{alg}/best_rmspbe_auc/{alg}.json'
        else:
            exp_path = f'experiments/stepsizes/{problem}/{alg}/best_rmspbe_auc/{alg}{stepsize}.json'

        generatePlot(ax, exp_path, bounds)

    lower = min(map(lambda x: x[0], bounds)) * 0.9
    upper = max(map(lambda x: x[1], bounds)) * 1.05

    if lower < 0.01:
        lower = -0.01

    # ax.set_ylim([lower, upper])
    ax.set_xlim([0, None])

    plt.show()
    exit()

    save_path = 'experiments/stepsizes/plots'
    os.makedirs(save_path, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/{name}_variance-curve_{stepsize}_{problem}.pdf', bbox_inches='tight', dpi=100)
