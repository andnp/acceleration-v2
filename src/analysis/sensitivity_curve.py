import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import tee

from src.analysis.results import getBestOverParameter, find, sliceOverParameter, getBestEnd, getBest

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

def getSensitivityData(results, param, reducer='best', overStream=None, bestBy='end'):
    useOtherStream = overStream is not None
    overStream = overStream if useOtherStream else results

    if reducer == 'best':
        bestStream = getBestOverParameter(overStream, param, bestBy=bestBy)

    elif reducer == 'slice':
        l, r = tee(overStream)
        if bestBy == 'end':
            best = getBestEnd(l)
        elif bestBy == 'auc':
            best = getBest(l)

        bestStream = sliceOverParameter(r, best, param)

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

    return x, y, e

def plotSensitivity(results, param, ax, reducer='best', stderr=True, overStream=None, color=None, label=None, dashed=False, bestBy='end'):
    x, y, e = getSensitivityData(results, param, reducer, overStream, bestBy)

    if dashed:
        dashes = ':'
    else:
        dashes = None

    ax.plot(x, y, label=label, linestyle=dashes, color=color, linewidth=2)
    if stderr:
        low_ci, high_ci = confidenceInterval(np.array(y), np.array(e))
        ax.fill_between(x, low_ci, high_ci, color=color, alpha=0.4)

    ax.legend()
    max_y = getMaxY(y)
    min_y = min(y) * .95

    return (min_y, max_y)

def sensitivityCurve(ax, x, y, e=None, color=None, alphaMain=1, label=None, dashed=False):
    if dashed:
        dashes = ':'
    else:
        dashes = None

    ax.plot(x, y, label=label, linestyle=dashes, color=color, alpha=alphaMain, linewidth=2)
    if e is not None:
        low_ci, high_ci = confidenceInterval(np.array(y), np.array(e))
        ax.fill_between(x, low_ci, high_ci, color=color, alpha=0.4 * alphaMain)

    max_y = getMaxY(y)
    min_y = min(y) * .95

    return (min_y, max_y)


def confidenceInterval(mean, stderr):
    stderr = stderr.clip(0, 1)
    # stderr = 0 if np.isnan(stderr) else stderr
    return (mean - stderr, mean + stderr)
