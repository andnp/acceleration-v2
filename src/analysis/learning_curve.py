import os
from src.analysis.results import getBest, getBestEnd
import matplotlib.pyplot as plt

def confidenceInterval(mean, stderr):
    return (mean - stderr, mean + stderr)

def save(exp, name):
    exp_name = exp.getExperimentName()
    save_path = f'experiments/{exp_name}'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/{name}.pdf')

def plot(results, ax, color=None, label=None, labelParams=None):
    best = getBestEnd(results)
    print(best.exp.agent, best.params)
    return plotBest(best, ax, color, label, labelParams=labelParams)


def plotBest(best, ax, color=None, label=None, alphaMain=None, stderr=True, labelParams=None):
    label = label if label is not None else best.exp.agent

    params = ''
    if labelParams is not None:
        l = [f'{key}-{best.params[key]}' for key in labelParams]
        params = ' ' + ' '.join(l)

    base, = ax.plot(best.mean(), label=label + params, color=color, alpha=alphaMain, linewidth=2)
    if stderr:
        (low_ci, high_ci) = confidenceInterval(best.mean(), best.stderr())
        ax.fill_between(range(best.mean().shape[0]), low_ci, high_ci, color=base.get_color(), alpha=0.4)
    ax.legend()
