import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.sensitivity_curve import plotSensitivity, save
from src.analysis.results import loadResults, whereParameterEquals, getBest, find, getBestEnd
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

# name = 'test'
# problems = ['SmallChainTabular5050', 'Boyan']

# name = 'policy'
# problems = ['SmallChainTabular5050', 'SmallChainTabular4060', 'Baird']

# name = 'features'
# problems = ['SmallChainTabular5050', 'SmallChainInverted5050', 'SmallChainDependent5050' 'Boyan']

name = 'all'
problems = ['SmallChainTabular5050LeftZero', 'SmallChainInverted5050LeftZero', 'SmallChainDependent5050LeftZero', 'SmallChainTabular5050', 'SmallChainTabular4060', 'SmallChainInverted5050', 'SmallChainInverted4060', 'SmallChainDependent5050', 'SmallChainDependent4060', 'Boyan', 'Baird']

algorithms = ['gtd2', 'tdc']
stepsizes = ['constant', 'adagrad', 'schedule']

def generatePlotTTA(ax, exp_paths, bestBy, bounds):
    ax.set_xscale("log", basex=2)
    for exp_path in exp_paths:
        if 'amsgrad' in exp_path:
            continue

        exp = loadExperiment(exp_path)
        rmsve = loadResults(exp, 'errors_summary.npy')
        rmspbe = loadResults(exp, 'rmspbe_summary.npy')

        color = colors[exp.agent]
        label = exp.agent

        b = plotSensitivity(rmsve, 'ratio', ax, overStream=rmspbe, color=color, label=label, bestBy=bestBy)
        bounds.append(b)

def generatePlotSTA(ax, exp_paths, bestBy, bounds):
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        rmsve = loadResults(exp, 'errors_summary.npy')
        rmspbe = loadResults(exp, 'rmspbe_summary.npy')

        color = colors[exp.agent]
        label = exp.agent

        if bestBy == 'end':
            metric = lambda m: np.mean(m[-int(m.shape[0] * .1):])
            best_rmspbe = getBestEnd(rmspbe)
            best = find(rmsve, best_rmspbe)
        elif bestBy == 'auc':
            metric = np.mean
            best_rmspbe = getBest(rmspbe)
            best = find(rmsve, best_rmspbe)

        m = metric(best.mean())
        ax.hlines(m, 2**-6, 2**6, color=color, label=label)

        bounds.append([m, m])

if __name__ == "__main__":
    f, axes = plt.subplots(len(stepsizes), len(problems) * 2)

    for i, ss in enumerate(stepsizes):
        for j, problem in enumerate(problems):
            bounds = []
            for alg in algorithms:
                if ss == 'constant':
                    exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/{alg}.json')
                    td_paths = [f'experiments/stepsizes/{problem}/td/td.json']
                else:
                    exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/{alg}{ss}.json')
                    td_paths = glob.glob(f'experiments/stepsizes/{problem}/td/td{ss}.json')


                generatePlotTTA(axes[i, 2 * j], exp_paths, 'auc', bounds)
                generatePlotSTA(axes[i, 2 * j], td_paths, 'auc', bounds)
                generatePlotSTA(axes[i, 2 * j + 1], td_paths, 'end', bounds)
                generatePlotTTA(axes[i, 2 * j + 1], exp_paths, 'end', bounds)

                lower = min(map(lambda x: x[0], bounds)) * 0.9
                upper = max(map(lambda x: x[1], bounds)) * 1.05

                if lower < 0.01:
                    lower = -0.01

                if upper > 100:
                    upper = 8

                if i == 0:
                    axes[i, 2 * j].set_title(f'{problem}\n{ss}')
                else:
                    axes[i, 2 * j].set_title(f'{ss}')

                axes[i, 2 * j].set_ylim([lower, upper])
                axes[i, 2 * j + 1].set_ylim([lower, upper])
                axes[i, 2 * j].axvline(1, linestyle=':', color='grey', linewidth=0.5)
                axes[i, 2 * j + 1].axvline(1, linestyle=':', color='grey', linewidth=0.5)


    # plt.show()
    # exit()

    save_path = 'experiments/stepsizes/plots'
    os.makedirs(save_path, exist_ok=True)

    width = len(problems) * 8
    height = len(stepsizes) * (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/ss_eta_{name}_rmsve-over-rmspbe.png', dpi=125)
