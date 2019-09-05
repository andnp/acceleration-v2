import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plot, save
from src.analysis.results import loadResults, whereParameterEquals
from src.utils.model import loadExperiment

from src.utils.path import fileName

def generatePlot(exp_paths):
    ax = plt.gca()
    # ax.semilogx()
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, 'rmspbe_summary.npy')

        label = fileName(exp_path).replace('.json', '')

        plot(results, ax, label=label, bestBy='auc')

    plt.show()
    # save(exp, f'learning-curve')
    plt.clf()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    tmp = loadExperiment(exp_paths[0])

    generatePlot(exp_paths)
