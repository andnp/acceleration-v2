import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from functools import partial
from multiprocessing.pool import Pool

from src.analysis.learning_curve import plot, save
from src.analysis.results import loadResults, whereParameterEquals
from src.utils.model import loadExperiment

trial = 0

def columnReducer(col):
    return lambda m: m[:, col]

def generatePlot(exp_paths):
    ax = plt.gca()

    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, 'tde_variance_summary.npy')

        plot(results, ax)

    # plt.show()

    exp_name = exp.getExperimentName()
    save_path = f'experiments/{exp_name}/trials/{trial}'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/tde_variance-curve.pdf')

if __name__ == "__main__":
    exp_paths = sys.argv[1:]

    generatePlot(exp_paths)
