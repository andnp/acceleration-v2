import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from functools import partial
from multiprocessing.pool import Pool

from src.analysis.sensitivity_curve import plotSensitivity, save
from src.analysis.results import loadResults, whereParameterEquals
from src.utils.model import loadExperiment

def generatePlot(exp_paths, lmbda):
    ax = plt.gca()

    bounds = []
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp)
        results = whereParameterEquals(results, 'lambda', lmbda)

        bound = plotSensitivity(results, 'ratio', ax)
        bounds.append(bound)

    lower = min(map(lambda x: x[0], bounds))
    upper = max(map(lambda x: x[1], bounds))

    ax.set_ylim([lower, upper])

    ax.set_xscale("log", basex=2)
    save(exp, f'beta-sensitivity_lambda-{lmbda}', trial = 0)
    plt.clf()
    print('done:', lmbda)

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    tmp = loadExperiment(exp_paths[0])
    lmbdas = tmp._d['metaParameters']['lambda']

    pool = Pool(len(lmbdas))
    pool.map(partial(generatePlot, exp_paths), lmbdas)

