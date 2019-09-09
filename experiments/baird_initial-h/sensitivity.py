import os
import sys
import glob
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.sensitivity_curve import plotSensitivity, save
from src.analysis.results import loadResults, whereParameterEquals
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.path import fileName

def generatePlot(exp_paths):
    ax = plt.gca()

    bounds = []
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp)

        if exp.agent == 'TDadagrad':
            continue

        color = colors[exp.agent]

        label = exp.agent.replace('adagrad', '')

        bound = plotSensitivity(results, 'h_variance', ax, label=label, color=color, bestBy='end')
        bounds.append(bound)

    # lower = min(map(lambda x: x[0], bounds))
    # upper = max(map(lambda x: x[1], bounds))

    # ax.set_ylim([lower, upper])

    ax.set_xscale("log", basex=2)
    plt.show()
    # save(exp, f'alpha-sensitivity')
    plt.clf()

if __name__ == "__main__":
    exp_paths = glob.glob('experiments/baird_initial-h/*.json')

    generatePlot(exp_paths)
