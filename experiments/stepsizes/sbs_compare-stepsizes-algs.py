import os
import sys
import glob
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import plot, save, plotBest
from src.analysis.results import loadResults, whereParameterGreaterEq, getBest, find
from src.analysis.colormap import stepsize_colors as colors
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

error = 'rmsve'

# name = 'policy'
# problems = ['SmallChainTabular5050', 'SmallChainTabular4060', 'Baird']

# name = 'features'
# problems = ['SmallChainTabular5050', 'SmallChainInverted5050', 'SmallChainDependent5050' 'Boyan']

name = 'all'
problems = ['SmallChainTabular5050LeftZero', 'SmallChainInverted5050LeftZero', 'SmallChainDependent5050LeftZero', 'SmallChainTabular5050', 'SmallChainTabular4060', 'SmallChainInverted5050', 'SmallChainInverted4060', 'SmallChainDependent5050', 'SmallChainDependent4060', 'SmallChainRandomCluster1090', 'SmallChainRandomCluster4060', 'SmallChainRandomCluster5050', 'SmallChainOuterRandomCluster1090', 'Boyan', 'Baird']

algorithms = ['gtd2', 'tdc']
stepsizes = ['constant', 'adagrad', 'amsgrad', 'schedule']

if error == 'rmsve':
    errorfile = 'errors_summary.npy'
elif error == 'rmspbe':
    errorfile = 'rmspbe_summary.npy'

def generatePlotTTA(ax, exp_paths, bestBy, bounds):
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, errorfile)

        color = colors[exp.agent]
        label = exp.agent

        dashed = False
        if 'TDC' in label:
            dashed = True

        results = whereParameterGreaterEq(results, 'ratio', 1)

        b = plot(results, ax, label=label, color=color, dashed=dashed, bestBy=bestBy)
        bounds.append(b)

if __name__ == "__main__":
    f, axes = plt.subplots(2, len(problems))

    for j, problem in enumerate(problems):
        bounds = []
        path = f'experiments/stepsizes/{problem}/lstd.json'
        lstd_exp = loadExperiment(path)
        LSTD_res = loadResults(lstd_exp, errorfile)

        LSTD_best = getBest(LSTD_res)

        b = plotBest(LSTD_best, axes[0, j], color=colors['LSTD'], label='LSTD', alphaMain=0.5)
        b = plotBest(LSTD_best, axes[1, j], color=colors['LSTD'], label='LSTD', alphaMain=0.5)
        bounds.append(b)

        for ss in stepsizes:
            for alg in algorithms:
                print(problem, alg, ss)
                if ss == 'constant':
                    exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/{alg}.json')
                else:
                    exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/{alg}{ss}.json')

                if len(exp_paths) == 0:
                    continue
                exp = loadExperiment(exp_paths[0])

                generatePlotTTA(axes[0, j], exp_paths, 'auc', bounds)
                generatePlotTTA(axes[1, j], exp_paths, 'end', bounds)

                lower = min(map(lambda x: x[0], bounds)) * 0.9
                upper = max(map(lambda x: x[1], bounds)) * 1.05

                if lower < 0.01:
                    lower = -0.01

                axes[0, j].set_title(f'{problem}\nauc')
                axes[1, j].set_title(f'end')

                axes[0, j].set_ylim([lower, upper])
                axes[1, j].set_ylim([lower, upper])


    # plt.show()
    # exit()

    save_path = 'experiments/stepsizes/plots'
    os.makedirs(save_path, exist_ok=True)

    width = len(problems) * 8
    height = 2 * (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/{name}_compare-stepsizes-algs_{error}.png', dpi=100)
