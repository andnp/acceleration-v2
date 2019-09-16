import os
import sys
import glob
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import plot, save
from src.analysis.results import loadResults, whereParameterEquals
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.path import fileName

def generatePlot(exp_paths):
    ax = plt.gca()
    # ax.semilogx()
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp)
        const, unconst = tee(results)

        const = whereParameterEquals(const, 'ratio', 1)

        color = colors[exp.agent]
        label = exp.agent

        plot(unconst, ax, label=label + '_unc', color=color, dashed=True)
        plot(const, ax, label=label, color=color, dashed=False)

    plt.show()
    # save(exp, f'rmsve_learning-curve')
    plt.clf()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]

    generatePlot(exp_paths)
