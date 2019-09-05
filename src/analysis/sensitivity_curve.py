import numpy as np
import os
import matplotlib.pyplot as plt

from src.analysis.results import getBestOverParameter

def save(exp, name):
    exp_name = exp.getExperimentName()
    save_path = f'experiments/{exp_name}'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/{name}.pdf')

def getMaxY(arr):
    m = arr[0]
    for y in arr:
        if np.isnan(y):
            continue

        if y > 1.5 * m:
            continue

        if y > m:
            m = y

    return m

def plotSensitivity(results, param, ax, label=None):
    best = getBestOverParameter(results, param)

    for x in best:
        res = best[x]
        print(res.exp.agent + ': ', res.params)

    x = sorted(list(best))
    y = [np.mean(best[k].mean()) for k in x]

    e = [np.mean(best[k].stderr()) for k in x]

    exp = best[x[0]].exp

    label = exp.agent if label is None else label

    ax.plot(x, y, label=label, linewidth=2)
    low_ci, high_ci = confidenceInterval(np.array(y), np.array(e))
    ax.fill_between(x, low_ci, high_ci, alpha=0.4)

    ax.legend()
    max_y = getMaxY(y)
    min_y = min(y) * .95

    return (min_y, max_y)

def confidenceInterval(mean, stderr):
    stderr = stderr.clip(0, 1)
    # stderr = 0 if np.isnan(stderr) else stderr
    return (mean - stderr, mean + stderr)
