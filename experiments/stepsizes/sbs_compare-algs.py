import os
import sys
import glob
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import plot, save, plotBest
from src.analysis.results import loadResults, whereParameterGreaterEq, getBest, find
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

error = 'rmsve'

# name = 'policy'
# problems = ['SmallChainTabular5050', 'SmallChainTabular4060', 'Baird']

# name = 'features'
# problems = ['SmallChainTabular5050', 'SmallChainInverted5050', 'SmallChainDependent5050' 'Boyan']

name = 'regh'
problems = ['SmallChainTabular5050', 'SmallChainTabular4060', 'SmallChainInverted5050', 'SmallChainInverted4060', 'SmallChainDependent5050', 'SmallChainDependent4060', 'Boyan', 'Baird']
algorithms = ['gtd2', 'tdc', 'td', 'regh_tdc']
stepsizes = ['constant', 'adagrad', 'schedule']

# name = 'all'
# problems = ['SmallChainTabular5050LeftZero', 'SmallChainInverted5050LeftZero', 'SmallChainDependent5050LeftZero', 'SmallChainTabular5050', 'SmallChainTabular4060', 'SmallChainInverted5050', 'SmallChainInverted4060', 'SmallChainDependent5050', 'SmallChainDependent4060', 'SmallChainRandomCluster1090', 'SmallChainRandomCluster4060', 'SmallChainRandomCluster5050', 'SmallChainOuterRandomCluster1090', 'Boyan', 'Baird']
# algorithms = ['gtd2', 'tdc', 'td']
# stepsizes = ['constant', 'adagrad', 'amsgrad', 'schedule']


if error == 'rmsve':
    errorfile = 'errors_summary.npy'
elif error == 'rmspbe':
    errorfile = 'rmspbe_summary.npy'

def generatePlotTTA(ax, exp_paths, bounds):
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, errorfile)
        const, unconst = tee(results)

        color = colors[exp.agent]
        label = exp.agent

        # if error == 'rmsve':
        #     rmspbe = loadResults(exp, 'rmspbe_summary.npy')
        #     rmspbe_unconst, rmspbe_const = tee(rmspbe)

        #     rmspbe_const = whereParameterGreaterEq(rmspbe_const, 'ratio', 1)

        #     best_rmspbe_unconst = getBest(rmspbe_unconst)
        #     best_rmspbe_const = getBest(rmspbe_const)

        #     best_unconst = find(unconst, best_rmspbe_unconst)
        #     best_const = find(const, best_rmspbe_const)

        # elif error == 'rmspbe':
        const = whereParameterGreaterEq(const, 'ratio', 1)
        best_unconst = getBest(unconst)
        best_const = getBest(const)

        if 'Regh' in label:
            print(best_const.params)


        b = plotBest(best_unconst, ax, label=label + '_unc', color=color, dashed=True)
        bounds.append(b)

        b = plotBest(best_const, ax, label=label, color=color, dashed=False)
        bounds.append(b)

def generatePlotSSA(ax, exp_paths, bounds):
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, errorfile)

        color = colors[exp.agent]
        label = exp.agent

        # if error == 'rmsve':
        #     rmspbe = loadResults(exp, 'rmspbe_summary.npy')
        #     best_rmspbe = getBest(rmspbe)

        #     best = find(results, best_rmspbe)

        # elif error == 'rmspbe':
        #     best = getBest(results)

        best = getBest(results)

        b = plotBest(best, ax, label=label, color=color, dashed=False)
        bounds.append(b)


if __name__ == "__main__":
    f, axes = plt.subplots(len(stepsizes), len(problems))

    for i, ss in enumerate(stepsizes):
        for j, problem in enumerate(problems):
            bounds = []

            path = f'experiments/stepsizes/{problem}/lstd.json'
            lstd_exp = loadExperiment(path)
            LSTD_res = loadResults(lstd_exp, errorfile)

            LSTD_best = getBest(LSTD_res)

            b = plotBest(LSTD_best, axes[i, j], color=colors['LSTD'], label='LSTD', alphaMain=0.5)
            bounds.append(b)

            for alg in algorithms:
                print(problem, alg, ss)
                if ss == 'constant':
                    exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/{alg}.json')
                else:
                    exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/{alg}{ss}.json')

                if len(exp_paths) == 0:
                    continue
                exp = loadExperiment(exp_paths[0])

                if '_h' in alg or alg == 'td':
                    generatePlotSSA(axes[i, j], exp_paths, bounds)
                else:
                    generatePlotTTA(axes[i, j], exp_paths, bounds)

                lower = min(map(lambda x: x[0], bounds)) * 0.9
                upper = max(map(lambda x: x[1], bounds)) * 1.05

                if lower < 0.01:
                    lower = -0.01

                if i == 0:
                    axes[i, j].set_title(f'{problem}\n{ss}')
                else:
                    axes[i, j].set_title(f'{ss}')

                axes[i, j].set_ylim([lower, upper])


    # plt.show()
    # exit()

    save_path = 'experiments/stepsizes/plots'
    os.makedirs(save_path, exist_ok=True)

    width = len(problems) * 8
    height = len(stepsizes) * (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/{name}_compare-algs_{error}.png', dpi=100)
