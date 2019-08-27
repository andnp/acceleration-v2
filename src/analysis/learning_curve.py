import os
from src.analysis.results import getBest
import matplotlib.pyplot as plt

def confidenceInterval(mean, stderr):
    return (mean - stderr, mean + stderr)

def save(exp, name, trial = 0):
    exp_name = exp.getExperimentName()
    save_path = f'experiments/{exp_name}/trials/{trial}'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/{name}.pdf')

def plot(results, ax):
    best = getBest(results)
    print(best.exp.agent, best.params)
    return plotBest(best, ax)


def plotBest(best, ax, color=None, label=None, alphaMain=None, stderr=True):
    label = label if label is not None else best.exp.agent

    base, = ax.plot(best.mean(), label=label, color=color, alpha=alphaMain, linewidth=2)
    if stderr:
        (low_ci, high_ci) = confidenceInterval(best.mean(), best.stderr())
        ax.fill_between(range(best.mean().shape[0]), low_ci, high_ci, color=base.get_color(), alpha=0.4)
    ax.legend()
