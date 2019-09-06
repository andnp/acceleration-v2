import os
import sys
import glob
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
        results = loadResults(exp)

        is_h_star = exp.getPermutation(0)['metaParameters']['use_ideal_h']
        agent = exp.agent.replace('adagrad', '')

        label = agent
        if is_h_star:
            label += '-h*'

        plot(results, ax, label=label)

    plt.show()
    # save(exp, f'learning-curve')
    plt.clf()

if __name__ == "__main__":
    exp_paths = glob.glob('experiments/idealH_baird/*.json')
    tmp = loadExperiment(exp_paths[0])

    generatePlot(exp_paths)
