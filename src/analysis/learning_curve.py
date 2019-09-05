import os
import numpy as np
from src.analysis.results import getBest, getBestEnd
import matplotlib.pyplot as plt

def confidenceInterval(mean, stderr):
    return (mean - stderr, mean + stderr)

def save(exp, name):
    exp_name = exp.getExperimentName()
    save_path = f'experiments/{exp_name}/plots'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/{name}.pdf')

def plot(results, ax, color=None, label=None, labelParams=None, bestBy='end', dashed=False):
    if bestBy == 'end':
        best = getBestEnd(results)
    elif bestBy == 'auc':
        best = getBest(results)
    else:
        raise Exception('I can only get best by "end" or "auc"')

    print(best.exp.agent, best.params)
    return plotBest(best, ax, color, label, labelParams=labelParams, dashed=dashed)


def plotBest(best, ax, color=None, label=None, alphaMain=None, stderr=True, labelParams=None, dashed=False):
    label = label if label is not None else best.exp.agent

    params = ''
    if labelParams is not None:
        l = [f'{key}-{best.params[key]}' for key in labelParams]
        params = ' ' + ' '.join(l)

    mean = best.mean()
    ste = best.stderr()

    if len(mean.shape) == 1:
        mean = np.reshape(mean, (-1, 1))
        ste = np.reshape(ste, (-1, 1))

    if type(label) != list:
        label = [label] * mean.shape[1]

    if dashed:
        dashes = ':'
    else:
        dashes = None

    for i in range(mean.shape[1]):
        base, = ax.plot(mean[:, i], linestyle=dashes, label=label[i] + params, color=color, alpha=alphaMain, linewidth=2)
        if stderr:
            (low_ci, high_ci) = confidenceInterval(mean[:, i], ste[:, i])
            ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color=base.get_color(), alpha=0.4)

    ax.legend()
