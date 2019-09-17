import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import plot, save, plotBest
from src.analysis.results import loadResults, whereParameterEquals, getBest, getBestEnd, find
from src.analysis.colormap import stepsize_colors as colors
from src.utils.model import loadExperiment
from src.utils.path import up
from src.utils.arrays import first

from src.utils.path import fileName

def generatePlot(exp_paths):
    f, axes = plt.subplots(2,2)

    # get LSTD solution
    path = up(up(first(exp_paths))) + '/lstd.json'
    exp = loadExperiment(path)
    LSTD_rmsve_results = loadResults(exp, 'errors_summary.npy')
    LSTD_rmspbe_results = loadResults(exp, 'rmspbe_summary.npy')

    LSTD_rmsve = getBest(LSTD_rmsve_results)
    LSTD_rmspbe = getBest(LSTD_rmspbe_results)

    rmspbe_bounds = []
    rmsve_bounds = []

    bounds = plotBest(LSTD_rmspbe, axes[0, 0], color=colors['LSTD'], label='LSTD', alphaMain=0.5)
    rmspbe_bounds.append(bounds)

    bounds = plotBest(LSTD_rmspbe, axes[0, 1], color=colors['LSTD'], label='LSTD', alphaMain=0.5)
    rmspbe_bounds.append(bounds)

    bounds = plotBest(LSTD_rmsve, axes[1, 0], color=colors['LSTD'], label='LSTD', alphaMain=0.5)
    rmsve_bounds.append(bounds)

    bounds = plotBest(LSTD_rmsve, axes[1, 1], color=colors['LSTD'], label='LSTD', alphaMain=0.5)
    rmsve_bounds.append(bounds)


    # RMSPBE plots
    ax = axes[0,0]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, 'rmspbe_summary.npy')

        color = colors[exp.agent]
        label = exp.agent

        bounds = plot(results, ax, label=label, color=color, dashed=False, bestBy='auc')
        rmspbe_bounds.append(bounds)

        ax.set_ylabel("RMSPBE")
        ax.set_title("RMSPBE")

    ax = axes[0,1]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        rmsve = loadResults(exp, 'errors_summary.npy')
        results = loadResults(exp, 'rmspbe_summary.npy')

        best = getBestEnd(rmsve)
        best_rmspbe = find(results, best)

        color = colors[exp.agent]
        label = exp.agent

        bounds = plotBest(best_rmspbe, ax, label=label, color=color, dashed=False)
        rmspbe_bounds.append(bounds)

        ax.set_title("RMSVE")

    # RMSVE plots
    ax = axes[1,0]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        rmsve = loadResults(exp, 'errors_summary.npy')
        rmspbe = loadResults(exp, 'rmspbe_summary.npy')

        # best PBE using AUC
        best = getBest(rmspbe)
        best_rmsve = find(rmsve, best)

        color = colors[exp.agent]
        label = exp.agent

        bounds = plotBest(best_rmsve, ax, label=label, color=color, dashed=False)
        rmsve_bounds.append(bounds)

        ax.set_ylabel("RMSVE")

    ax = axes[1,1]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp)

        color = colors[exp.agent]
        label = exp.agent

        bounds = plot(results, ax, label=label, color=color, dashed=False, bestBy='end')
        rmsve_bounds.append(bounds)


    # rmspbe
    rmspbe_lower = min(map(lambda x: x[0], rmspbe_bounds)) * 0.9
    rmspbe_upper = max(map(lambda x: x[1], rmspbe_bounds)) * 1.05

    if rmspbe_lower < 0.01:
        rmspbe_lower = -0.01

    axes[0, 0].set_ylim([rmspbe_lower, rmspbe_upper])
    axes[0, 1].set_ylim([rmspbe_lower, rmspbe_upper])

    # rmsve
    rmsve_lower = min(map(lambda x: x[0], rmsve_bounds)) * 0.9
    rmsve_upper = max(map(lambda x: x[1], rmsve_bounds)) * 1.05

    if rmsve_lower < 0.01:
        rmsve_lower = -0.01

    axes[1, 0].set_ylim([rmsve_lower, rmsve_upper])
    axes[1, 1].set_ylim([rmsve_lower, rmsve_upper])

    save(exp, f'rmsve_rmspbe_square', type='svg')
    # plt.show()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]

    generatePlot(exp_paths)
