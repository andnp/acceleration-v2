import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import plotBest, save
from src.analysis.results import loadResults, whereParameterEquals, getBest, find, getBestEnd
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

name = 'all'
problems = ['SmallChainTabular5050', 'SmallChainTabular4060', 'SmallChainInverted5050', 'SmallChainInverted4060', 'SmallChainDependent5050', 'SmallChainDependent4060', 'Boyan', 'Baird']

algorithms = ['gtd2', 'tdc', 'td']
stepsizes = ['constant']
error = 'rmspbe'

def generatePlot(ax, exp_paths, bounds):
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        variance = loadResults(exp, 'expupd_summary.npy')

        agent = exp.agent
        color = colors[agent]
        label = agent

        best = getBest(variance)
        if 'GTD2' in agent or 'TDC' in agent:
            best.reducer(lambda m: m[:, 1])
        else:
            best.reducer(lambda m: m[:, 1])

        b = plotBest(best, ax, color=color, label=label)
        bounds.append(b)

if __name__ == "__main__":
    f, axes = plt.subplots(len(stepsizes), len(problems))

    for i, ss in enumerate(stepsizes):
        for j, problem in enumerate(problems):
            bounds = []

            for alg in algorithms:
                if ss == 'constant':
                    exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/best_rmspbe_auc/{alg}.json')
                else:
                    exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/best_rmspbe_auc/{alg}{ss}.json')


                generatePlot(axes[j], exp_paths, bounds)

            # ----------------------
            # -- Set y-axis bounds -
            # ----------------------

            if i == 0:
                axes[j].set_title(f'{problem}\n{ss}')
            else:
                axes[j].set_title(f'{ss}')

            if problem == 'Baird':
                axes[j].set_ylim([0, 0.5])

    # plt.show()
    # exit()

    save_path = 'experiments/stepsizes/plots'
    os.makedirs(save_path, exist_ok=True)

    width = len(problems) * 8
    height = len(stepsizes) * (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/{name}_expupd_{error}', dpi=100)
