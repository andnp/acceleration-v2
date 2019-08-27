import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from functools import partial
from multiprocessing.pool import Pool

from src.analysis.learning_curve import plot, save
from src.analysis.results import loadResults, whereParameterEquals
from src.utils.model import loadExperiment

def generatePlot(exp_paths, lmbda):
    ax = plt.gca()
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp)
        results = whereParameterEquals(results, 'lambda', lmbda)

        plot(results, ax)

    # plt.show()
    save(exp, f'learning-curve_lambda-{lmbda}', trial=0)
    plt.clf()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    tmp = loadExperiment(exp_paths[0])
    lmbdas = tmp._d['metaParameters']['lambda']

    pool = Pool(len(lmbdas))
    pool.map(partial(generatePlot, exp_paths), lmbdas)
