import os
import sys
import glob
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import plot, save, plotBest
from src.analysis.results import loadResults, whereParameterEquals, getBest
from src.analysis.colormap import stepsize_colors as colors
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

error = 'rmspbe'

# name = 'policy'
# problems = ['SmallChainTabular5050', 'SmallChainTabular4060', 'Baird']

# name = 'features'
# problems = ['SmallChainTabular5050', 'SmallChainInverted5050', 'Boyan']

name = 'all'
problems = ['SmallChainTabular5050', 'SmallChainTabular4060', 'SmallChainInverted5050', 'SmallChainInverted4060', 'Boyan', 'Baird']

algorithms = ['gtd2', 'gtd2_h', 'tdc', 'tdc_h', 'td']

if error == 'rmsve':
    errorfile = 'errors_summary.npy'
    bestBy = 'end'
elif error == 'rmspbe':
    errorfile = 'rmspbe_summary.npy'
    bestBy = 'auc'

def generatePlotTTA(ax, exp_paths, bounds):
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, errorfile)
        const, unconst = tee(results)

        const = whereParameterEquals(const, 'ratio', 1)

        color = colors[exp.agent]
        label = exp.agent

        b = plot(unconst, ax, label=label + '_unc', color=color, dashed=True, bestBy=bestBy)
        bounds.append(b)

        b = plot(const, ax, label=label, color=color, dashed=False, bestBy=bestBy)
        bounds.append(b)

def generatePlotSSA(ax, exp_paths, bounds):
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, errorfile)

        color = colors[exp.agent]
        label = exp.agent

        b = plot(results, ax, label=label, color=color, dashed=False, bestBy=bestBy)
        bounds.append(b)


if __name__ == "__main__":
    f, axes = plt.subplots(len(algorithms), len(problems))

    for i, alg in enumerate(algorithms):
        for j, problem in enumerate(problems):
            bounds = []

            exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/*.json')
            exp = loadExperiment(exp_paths[0])

            path = up(up(first(exp_paths))) + '/lstd.json'
            lstd_exp = loadExperiment(path)
            LSTD_res = loadResults(lstd_exp, errorfile)

            LSTD_best = getBest(LSTD_res)

            b = plotBest(LSTD_best, axes[i, j], color=colors['LSTD'], label='LSTD', alphaMain=0.5)
            bounds.append(b)

            if '_h' in alg or alg == 'td':
                generatePlotSSA(axes[i, j], exp_paths, bounds)
            else:
                generatePlotTTA(axes[i, j], exp_paths, bounds)

            lower = min(map(lambda x: x[0], bounds)) * 0.9
            upper = max(map(lambda x: x[1], bounds)) * 1.05

            if lower < 0.01:
                lower = -0.01

            if i == 0:
                axes[i, j].set_title(f'{problem}\n{alg}')
            else:
                axes[i, j].set_title(f'{alg}')

            axes[i, j].set_ylim([lower, upper])


    # plt.show()

    save_path = 'experiments/stepsizes/plots'
    os.makedirs(save_path, exist_ok=True)

    width = len(problems) * 8
    f.set_size_inches((width, 24), forward=False)
    plt.savefig(f'{save_path}/{name}_{error}.png', dpi=250)
