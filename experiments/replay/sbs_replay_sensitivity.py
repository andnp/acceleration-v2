import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.sensitivity_curve import plotSensitivity
from src.analysis.results import loadResults
from src.analysis.colormap import colors
from src.utils.model import loadExperiment
from src.analysis.results import loadResults, whereParameterGreaterEq, whereParameterEquals, getBest, getBestEnd, find

error = 'rmspbe'

bestBy = lambda m: np.mean(m[0 : int(m.shape[0] * .2)])
# bestBy = 'end' # or 'auc'

name = 'paper'
problems = ['SmallChainTabular4060', 'SmallChainInverted4060', 'SmallChainDependent4060', 'Boyan', 'Baird']
algorithms = ['gtd2', 'tdc', 'td', 'tdrc', 'htd', 'vtrace']

stepsizes = ['constant', 'adagrad', 'amsgrad']

if error == 'rmsve':
    errorfile = 'errors_summary.npy'
elif error == 'rmspbe':
    errorfile = 'rmspbe_summary.npy'

def generatePlotTTA(ax, exp_paths, bestBy, bounds):
    ax.set_xscale("log", basex=2)
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, errorfile)

        agent = exp.agent
        color = colors[agent]
        label = agent

        if exp.agent == 'ReghTDC':
            results = whereParameterGreaterEq(results, 'ratio', 1.0)

        # reducer='best' chooses the best value of other parameters *per value of 'replay'*
        # reducer='slice' first chooses the best parameter setting, then sweeps over 'replay' with other parameters fixed
        b = plotSensitivity(results, 'replay', ax, reducer='best', color=color, label=label, bestBy=bestBy)
        bounds.append(b)

if __name__ == "__main__":
    f, axes = plt.subplots(len(stepsizes), len(problems))

    for i, ss in enumerate(stepsizes):
        for j, problem in enumerate(problems):
            bounds = []

            # --------------------
            # -- Plot other algs -
            # --------------------

            for alg in algorithms:
                print(ss, problem, alg)
                if ss == 'constant':
                    exp_paths = glob.glob(f'experiments/replay/{problem}/replay_{alg}/{alg}.json')
                else:
                    exp_paths = glob.glob(f'experiments/replay/{problem}/replay_{alg}/{alg}{ss}.json')


                generatePlotTTA(axes[i, j], exp_paths, bestBy, bounds)

            # ----------------------
            # -- Set y-axis bounds -
            # ----------------------

            lower = min(map(lambda x: x[0], bounds)) * 0.9
            upper = max(map(lambda x: x[1], bounds)) * 1.05

            if lower < 0.01:
                lower = -0.01

            if upper > 100:
                upper = 8

            if i == 0:
                axes[i, j].set_title(f'{problem}\n{ss}\nauc')
            else:
                axes[i, j].set_title(f'{ss} - auc')

            # axes[i, 2 * j].set_ylim([lower, upper])
            # axes[i, 2 * j + 1].set_ylim([lower, upper])

            if problem == 'Baird':
                axes[i, j].set_ylim([0.03, 2.0])


    # plt.show()
    # exit()

    save_path = 'experiments/replay/plots'
    os.makedirs(save_path, exist_ok=True)

    width = len(problems) * 8
    height = len(stepsizes) * (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/sbs-replay-sensitivity_{name}_{error}.png', dpi=100)
