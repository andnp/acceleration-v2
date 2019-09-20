import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import tee

from src.analysis.results import getBestOverParameter, find

def save(exp, name, type='pdf'):
    exp_name = exp.getExperimentName()
    save_path = f'experiments/{exp_name}/plots'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/{name}.{type}')

def getMaxY(arr):
    m = arr[0]
    m0 = m
    for y in arr:
        if np.isnan(y):
            return m0

        if y > 1.5 * m:
            continue

        if y > m:
            m = y

    return m

def plotSensitivity(results, param, ax, overStream=None, color=None, label=None, dashed=False, bestBy='end'):
    useOtherStream = overStream is not None
    overStream = overStream if useOtherStream else results
    bestStream = getBestOverParameter(overStream, param, bestBy=bestBy)

    x = sorted(list(bestStream))
    if useOtherStream:
        best = {}
        teed = tee(results, len(x))
        for i, k in enumerate(x):
            best[k] = find(teed[i], bestStream[k])

    else:
        best = bestStream


    if bestBy == 'end':
        metric = lambda m: np.mean(m[-int(m.shape[0] * .1):])
    elif bestBy == 'auc':
        metric = np.mean

    y = [metric(best[k].mean()) for k in x]
    e = [metric(best[k].stderr()) for k in x]

    exp = best[x[0]].exp

    label = label if label is not None else exp.agent
    if dashed:
        dashes = ':'
    else:
        dashes = None

    ax.plot(x, y, label=label, linestyle=dashes, color=color, linewidth=2)
    low_ci, high_ci = confidenceInterval(np.array(y), np.array(e))
    ax.fill_between(x, low_ci, high_ci, color=color, alpha=0.4)

    ax.legend()
    max_y = getMaxY(y)
    min_y = min(y) * .95

    return (min_y, max_y)

def confidenceInterval(mean, stderr):
    stderr = stderr.clip(0, 1)
    # stderr = 0 if np.isnan(stderr) else stderr
    return (mean - stderr, mean + stderr)
