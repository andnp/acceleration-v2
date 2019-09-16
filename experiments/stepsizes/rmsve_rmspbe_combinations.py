import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import plot, save, plotBest
from src.analysis.results import loadResults, whereParameterEquals, getBest, getBestEnd, find
from src.analysis.colormap import stepsize_colors as colors
from src.utils.model import loadExperiment

from src.utils.path import fileName

def generatePlot(exp_paths):
    f, axes = plt.subplots(2,2)

    # RMSPBE plots
    ax = axes[0,0]
    rmspbe_bounds = []
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, 'rmspbe_summary.npy')
        const, unconst = tee(results)

        const = whereParameterEquals(const, 'ratio', 1)

        color = colors[exp.agent]
        label = exp.agent

        bounds = plot(const, ax, label=label, color=color, dashed=False, bestBy='auc')
        rmspbe_bounds.append(bounds)

        bounds = plot(unconst, ax, label=label + '_unc', color=color, dashed=True, bestBy='auc')
        rmspbe_bounds.append(bounds)

        ax.set_ylabel("RMSPBE")
        ax.set_title("RMSPBE")

    ax = axes[0,1]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        rmsve = loadResults(exp, 'errors_summary.npy')
        results = loadResults(exp, 'rmspbe_summary.npy')
        const, unconst = tee(rmsve)
        const_res, unconst_res = tee(results)

        const = whereParameterEquals(const, 'ratio', 1)

        best = getBestEnd(const)
        best_unc = getBestEnd(unconst)
        best_rmspbe = find(const_res, best)
        best_rmspbe_unc = find(unconst_res, best_unc)

        color = colors[exp.agent]
        label = exp.agent

        bounds = plotBest(best_rmspbe, ax, label=label, color=color, dashed=False)
        rmspbe_bounds.append(bounds)

        bounds = plotBest(best_rmspbe_unc, ax, label=label + '_unc', color=color, dashed=True)
        rmspbe_bounds.append(bounds)

        ax.set_title("RMSVE")

    # RMSVE plots
    ax = axes[1,0]
    rmsve_bounds = []
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        rmsve = loadResults(exp, 'errors_summary.npy')
        rmspbe = loadResults(exp, 'rmspbe_summary.npy')
        const, unconst = tee(rmspbe)
        const_res, unconst_res = tee(rmsve)

        const = whereParameterEquals(const, 'ratio', 1)

        # best PBE using AUC
        best = getBest(const)
        best_unc = getBest(unconst)
        best_rmsve = find(const_res, best)
        best_rmsve_unc = find(unconst_res, best_unc)

        color = colors[exp.agent]
        label = exp.agent

        bounds = plotBest(best_rmsve, ax, label=label, color=color, dashed=False)
        rmsve_bounds.append(bounds)

        bounds = plotBest(best_rmsve_unc, ax, label=label + '_unc', color=color, dashed=True)
        rmsve_bounds.append(bounds)

        ax.set_ylabel("RMSVE")

    ax = axes[1,1]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp)
        const, unconst = tee(results)

        const = whereParameterEquals(const, 'ratio', 1)

        color = colors[exp.agent]
        label = exp.agent

        bounds = plot(const, ax, label=label, color=color, dashed=False, bestBy='end')
        rmsve_bounds.append(bounds)

        bounds = plot(unconst, ax, label=label + '_unc', color=color, dashed=True, bestBy='end')
        rmsve_bounds.append(bounds)

    # rmspbe
    rmspbe_lower = min(map(lambda x: x[0], rmspbe_bounds)) * 0.9
    rmspbe_upper = max(map(lambda x: x[1], rmspbe_bounds)) * 1.05
    axes[0, 0].set_ylim([rmspbe_lower, rmspbe_upper])
    axes[0, 1].set_ylim([rmspbe_lower, rmspbe_upper])

    # rmsve
    rmsve_lower = min(map(lambda x: x[0], rmsve_bounds)) * 0.9
    rmsve_upper = max(map(lambda x: x[1], rmsve_bounds)) * 1.05
    axes[1, 0].set_ylim([rmsve_lower, rmsve_upper])
    axes[1, 1].set_ylim([rmsve_lower, rmsve_upper])

    save(exp, f'rmsve_rmspbe_square')
    # plt.show()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    tmp = loadExperiment(exp_paths[0])

    generatePlot(exp_paths)
