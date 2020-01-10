import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.getcwd())

from src.analysis.results import loadResults, whereParameterEquals
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.path import fileName

error = 'rmspbe'

name = 'all'
problems = ['SmallChainTabular5050LeftZero', 'SmallChainInverted5050LeftZero', 'SmallChainDependent5050LeftZero', 'SmallChainTabular5050', 'SmallChainTabular4060', 'SmallChainInverted5050', 'SmallChainInverted4060', 'SmallChainDependent5050', 'SmallChainDependent4060', 'SmallChainRandomCluster1090', 'SmallChainRandomCluster4060', 'SmallChainRandomCluster5050', 'SmallChainOuterRandomCluster1090', 'Boyan', 'Baird']
algorithms = ['tdc','regh_tdc']
stepsizes = ['constant', 'adagrad', 'schedule']

# how far above results should "diverged" results go?
DIVERGENCE_MULTIPLIER = 1.2
# how much space should there be between algorithm "lanes"
ALG_WIDTH = 1
# how wide should the lanes be?
LANE_WIDTH = 0.25

if error == 'rmsve':
    errorfile = 'errors_summary.npy'
elif error == 'rmspbe':
    errorfile = 'rmspbe_summary.npy'

save_path = 'experiments/stepsizes/plots'
diverged_file = open(f'{save_path}/{name}_{error}_waterfall_diverged.txt', 'w')

def generatePlot(ax, exp_paths, ss, problem):
    # load results
    all_performance = {}
    for exp_path in exp_paths:
        try:
            exp = loadExperiment(exp_path)
        except:
            continue

        results = loadResults(exp, errorfile)
        results = whereParameterEquals(results, 'reg_h', 0.8)

        color = colors[exp.agent]
        label = exp.agent

        performance = []
        for r in results:
            curve = r.mean()
            m = np.mean(curve)

            # diverged if result doesn't exist
            if np.isscalar(curve):
                m = np.nan

            # diverged if the end of the curve is higher than the start
            elif curve[0] < curve[curve.shape[0] - 1]:
                m = np.nan

            # diverged if mean is larger than start
            # elif curve[0] < m:
            #     m = np.nan

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

        num_diverged = sum(np.isnan(performance))
        num_total = performance.shape[0]

        diverged_file.write(f'ss: {ss}; problem: {problem}; alg: {label}; perc: {num_diverged / num_total}; div: {num_diverged}; total: {num_total}\n')

        performance[np.isnan(performance)] = global_max * DIVERGENCE_MULTIPLIER
        ax.scatter([x_offset] * performance.shape[0] + np.random.uniform(-LANE_WIDTH, LANE_WIDTH, performance.shape[0]), performance, marker='o', facecolors='none', color=color)

        x_ticks.append(x_offset)
        x_labels.append(label)

        x_offset += ALG_WIDTH + 2 * LANE_WIDTH

    ax.xaxis.set_ticks(x_ticks)
    ax.set_xticklabels(x_labels)

if __name__ == "__main__":
    f, axes = plt.subplots(len(stepsizes), len(problems))
    for i, ss in enumerate(stepsizes):
        for j, problem in enumerate(problems):
            print(ss, problem)

            exp_paths = []
            for alg in algorithms:
                if ss == 'constant':
                    exp_paths.append(f'experiments/stepsizes/{problem}/{alg}/{alg}.json')
                else:
                    exp_paths.append(f'experiments/stepsizes/{problem}/{alg}/{alg}{ss}.json')


            generatePlot(axes[i, j], exp_paths, ss, problem)

            if i == 0:
                axes[i, j].set_title(f'{problem}\n{ss}')
            else:
                axes[i, j].set_title(f'{ss}')



    # plt.show()
    # exit()

    save_path = 'experiments/stepsizes/plots'
    os.makedirs(save_path, exist_ok=True)

    width = len(problems) * 8
    height = len(stepsizes) * (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/{name}_{error}_waterfall.png', dpi=100)
