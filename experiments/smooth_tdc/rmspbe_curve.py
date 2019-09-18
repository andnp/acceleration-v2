import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plot, save
from src.analysis.results import loadResults, whereParameterEquals, getBestEnd, find, splitOverParameter
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.path import fileName, up

def generatePlot(exp_paths):
    ax = plt.gca()
    # ax.semilogx()
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, 'rmspbe_summary.npy')

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)
        dashed = use_ideal_h

        label = exp.agent
        if use_ideal_h:
            label += '-h*'

        if use_ideal_h:
            continue

        if 'SmoothTDC' in exp.agent:
            agents = splitOverParameter(results, 'averageType')
            smooth_colors = {
                'ema': 'pink',
                'buffer': 'grey',
                'window': 'black',
            }
            plot(agents['ema'], ax, label=label + '_ema', bestBy='auc', color=smooth_colors['ema'], dashed=dashed)
            plot(agents['buffer'], ax, label=label + '_buffer', bestBy='auc', color=smooth_colors['buffer'], dashed=dashed)
            plot(agents['window'], ax, label=label + '_window', bestBy='auc', color=smooth_colors['window'], dashed=dashed)

        else:
            color = colors[exp.agent]
            plot(results, ax, label=label, bestBy='auc', color=color, dashed=dashed)

    # plt.show()
    # save(exp, f'rmspbe', type='png')
    problem = fileName(exp.getExperimentName())
    save_path = f'plots/'
    os.makedirs(save_path, exist_ok=True)

    fig = plt.gcf()
    fig.set_size_inches((13, 12), forward=False)
    plt.savefig(f'{save_path}/{problem}_rmspbe.png')

if __name__ == "__main__":
    exp_paths = sys.argv[1:]

    generatePlot(exp_paths)
