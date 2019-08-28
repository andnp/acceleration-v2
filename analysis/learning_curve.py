import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plot, save
from src.analysis.results import loadResults
from src.utils.model import loadExperiment

def generatePlot(exp_paths):
    ax = plt.gca()
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp)

        plot(results, ax)

    plt.show()
    # save(exp, f'learning-curve')
    plt.clf()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    tmp = loadExperiment(exp_paths[0])

    generatePlot(exp_paths)
