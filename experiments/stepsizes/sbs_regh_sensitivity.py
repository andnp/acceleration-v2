import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.sensitivity_curve import plotSensitivity, save
from src.analysis.results import loadResults, whereParameterEquals, getBest, find, getBestEnd
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

error = 'rmspbe'

# name = 'test'
# problems = ['SmallChainTabular5050', 'Boyan']

# name = 'policy'
# problems = ['SmallChainTabular5050', 'SmallChainTabular4060', 'Baird']

# name = 'features'
# problems = ['SmallChainTabular5050', 'SmallChainInverted5050', 'SmallChainDependent5050' 'Boyan']

name = 'regh'
problems = ['SmallChainTabular5050', 'SmallChainTabular4060', 'SmallChainInverted5050', 'SmallChainInverted4060', 'SmallChainDependent5050', 'SmallChainDependent4060', 'Boyan', 'Baird']
algorithms = ['regh_tdc']

# name = 'all'
# problems = ['SmallChainTabular5050LeftZero', 'SmallChainInverted5050LeftZero', 'SmallChainDependent5050LeftZero', 'SmallChainTabular5050', 'SmallChainTabular4060', 'SmallChainInverted5050', 'SmallChainInverted4060', 'SmallChainDependent5050', 'SmallChainDependent4060', 'SmallChainRandomCluster1090', 'SmallChainRandomCluster4060', 'SmallChainRandomCluster5050', 'SmallChainOuterRandomCluster1090', 'Boyan', 'Baird']
# algorithms = ['gtd2', 'tdc', 'tdc_ema_x']

stepsizes = ['constant', 'adagrad', 'schedule']

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

        b = plotSensitivity(results, 'reg_h', ax, reducer='best', color=color, label=label, bestBy=bestBy)
        bounds.append(b)

if __name__ == "__main__":
    f, axes = plt.subplots(len(stepsizes), len(problems) * 2)

    for i, ss in enumerate(stepsizes):
        for j, problem in enumerate(problems):
            bounds = []

            # --------------------
            # -- Plot other algs -
            # --------------------

            for alg in algorithms:
                print(ss, problem, alg)
                if ss == 'constant':
                    exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/{alg}.json')
                else:
                    exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/{alg}{ss}.json')


                generatePlotTTA(axes[i, 2 * j], exp_paths, 'auc', bounds)
                generatePlotTTA(axes[i, 2 * j + 1], exp_paths, 'end', bounds)

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
                axes[i, 2 * j].set_title(f'{problem}\n{ss}\nauc')
            else:
                axes[i, 2 * j].set_title(f'{ss} - auc')

            axes[i, 2 * j + 1].set_title(f'{ss} - end')

            # axes[i, 2 * j].set_ylim([lower, upper])
            # axes[i, 2 * j + 1].set_ylim([lower, upper])


    # plt.show()
    # exit()

    save_path = 'experiments/stepsizes/plots'
    os.makedirs(save_path, exist_ok=True)

    width = len(problems) * 8
    height = len(stepsizes) * (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/regh_{name}_{error}.png', dpi=100)
