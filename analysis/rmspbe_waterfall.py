import os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plot, save
from src.analysis.results import loadResults, whereParameterEquals
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.path import fileName

# how far above results should "diverged" results go?
DIVERGENCE_MULTIPLIER = 1.2
# how much space should there be between algorithm "lanes"
ALG_WIDTH = 1
# how wide should the lanes be?
LANE_WIDTH = 0.25
# above what value is a result considered "diverged"
MAX_ACCEPTABLE = 20

def generatePlot(exp_paths):
    ax = plt.gca()
    # load results
    all_performance = {}
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, 'rmspbe_summary.npy')

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)
        color = colors[exp.agent]

        label = exp.agent.replace('adagrad', '')
        if use_ideal_h:
            label += '-h*'

        performance = []
        for r in results:
            m = np.mean(r.mean())
            if m > MAX_ACCEPTABLE:
                m = np.nan
            performance.append(m)

        performance = np.array(performance)

        all_performance[label] = { 'res': performance, 'color': color }

    # find max among all algorithms
    global_max = -np.inf
    for key in all_performance:
        data = all_performance[key]
        performance = data['res']

        local_max = np.nanmax(performance)
        if local_max > global_max:
            global_max = local_max

    # plot results
    x_offset = 0.5
    x_ticks = []
    x_labels = []
    for key in all_performance:
        data = all_performance[key]

        label = key
        performance = data['res']
        color = data['color']

        performance[np.isnan(performance)] = global_max * DIVERGENCE_MULTIPLIER
        ax.scatter([x_offset] * performance.shape[0] + np.random.uniform(-LANE_WIDTH, LANE_WIDTH, performance.shape[0]), performance, marker='o', facecolors='none', color=color)

        x_ticks.append(x_offset)
        x_labels.append(label)

        x_offset += ALG_WIDTH + 2 * LANE_WIDTH

    ax.xaxis.set_ticks(x_ticks)
    ax.set_xticklabels(x_labels)

    plt.show()
    # save(exp, f'rmsve_waterfall')
    plt.clf()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    tmp = loadExperiment(exp_paths[0])

    generatePlot(exp_paths)
