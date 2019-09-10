import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plot, save, plotBest
from src.analysis.results import loadResults, whereParameterEquals, getBestEnd, find
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.path import fileName

def generatePlot(exp_paths):
    f, axes = plt.subplots(2,2)
    # ax.semilogx()
    ax = axes[0,0]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, 'rmspbe_summary.npy')

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)
        dashed = use_ideal_h
        color = colors[exp.agent]

        label = exp.agent.replace('adagrad', '')
        if use_ideal_h:
            label += '-h*'

        plot(results, ax, label=label, color=color, dashed=dashed)
        ax.set_ylabel("RMSPBE")
        ax.set_title("RMSPBE")

    ax = axes[0,1]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        rmsve = loadResults(exp, 'errors_summary.npy')
        results = loadResults(exp, 'rmspbe_summary.npy')

        best = getBestEnd(rmsve)
        best_rmspbe = find(results, best)

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)
        dashed = use_ideal_h
        color = colors[exp.agent]

        label = exp.agent.replace('adagrad', '')
        if use_ideal_h:
            label += '-h*'

        plotBest(best_rmspbe, ax, label=label, color=color, dashed=dashed)
        ax.set_title("RMSVE")

    ax = axes[1,0]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        rmsve = loadResults(exp, 'errors_summary.npy')
        rmspbe = loadResults(exp, 'rmspbe_summary.npy')

        # if exp.agent == 'TDadagrad':
        #     continue

        # best PBE using AUC
        best = getBestEnd(rmspbe)
        best_rmsve = find(rmsve, best)

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)
        dashed = use_ideal_h
        color = colors[exp.agent]

        label = exp.agent.replace('adagrad', '')
        if use_ideal_h:
            label += '-h*'

        plotBest(best_rmsve, ax, label=label, color=color, dashed=dashed)
        ax.set_ylabel("RMSVE")

    ax = axes[1,1]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp)

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)
        dashed = use_ideal_h
        color = colors[exp.agent]

        label = exp.agent.replace('adagrad', '')
        if use_ideal_h:
            label += '-h*'

        plot(results, ax, label=label, color=color, dashed=dashed)

    plt.show()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    tmp = loadExperiment(exp_paths[0])

    generatePlot(exp_paths)
