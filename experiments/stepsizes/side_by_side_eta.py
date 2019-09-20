import os
import sys
import glob
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.sensitivity_curve import plotSensitivity, save
from src.analysis.results import loadResults, whereParameterEquals, getBest, find
from src.analysis.colormap import stepsize_colors as colors
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

error = 'rmspbe'

name = 'test'
problems = ['SmallChainTabular5050', 'Boyan']

# name = 'policy'
# problems = ['SmallChainTabular5050', 'SmallChainTabular4060', 'Baird']

# name = 'features'
# problems = ['SmallChainTabular5050', 'SmallChainInverted5050', 'SmallChainDependent5050' 'Boyan']

# name = 'all'
# problems = ['SmallChainTabular5050LeftZero', 'SmallChainInverted5050LeftZero', 'SmallChainDependent5050LeftZero', 'SmallChainTabular5050', 'SmallChainTabular4060', 'SmallChainInverted5050', 'SmallChainInverted4060', 'SmallChainDependent5050', 'SmallChainDependent4060', 'Boyan', 'Baird']

algorithms = ['gtd2', 'tdc']

if error == 'rmsve':
    errorfile = 'errors_summary.npy'
elif error == 'rmspbe':
    errorfile = 'rmspbe_summary.npy'

def generatePlotTTA(ax, exp_paths, bounds):
    ax.set_xscale("log", basex=2)
    for exp_path in exp_paths:
        if 'amsgrad' in exp_path:
            continue

        exp = loadExperiment(exp_path)
        results = loadResults(exp, errorfile)

        color = colors[exp.agent]
        label = exp.agent

        b = plotSensitivity(results, 'ratio', ax, color=color, label=label, bestBy='auc')
        bounds.append(b)

if __name__ == "__main__":
    f, axes = plt.subplots(len(algorithms), len(problems))

    for i, alg in enumerate(algorithms):
        for j, problem in enumerate(problems):
            bounds = []

            exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/*.json')

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


    plt.show()
    exit()

    save_path = 'experiments/stepsizes/plots'
    os.makedirs(save_path, exist_ok=True)

    width = len(problems) * 8
    height = len(algorithms) * (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/eta_{name}_{error}.png', dpi=125)
