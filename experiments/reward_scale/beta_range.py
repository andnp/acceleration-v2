import os
import sys
import glob
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import plot, save, plotBest
from src.analysis.results import loadResults, whereParameterGreaterEq, whereParameterEquals, splitOverParameter, getBest, find
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

error = 'rmspbe'

problem = 'Boyan'

bestBy = 'auc'
MAX = 0.4

SMALL = 8
MEDIUM = 16
BIGGER = 20

plt.rc('font', size=SMALL)          # controls default text sizes
plt.rc('axes', titlesize=SMALL)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM)    # legend fontsize
plt.rc('figure', titlesize=BIGGER)  # fontsize of the figure title

if error == 'rmsve':
    errorfile = 'errors_summary.npy'
elif error == 'rmspbe':
    errorfile = 'rmspbe_summary.npy'

def bound(x, mi, ma):
    return np.min((np.max((x, mi)), ma))

if __name__ == "__main__":
    ax = plt.gca()
    f = plt.gcf()

    td_exp = loadExperiment(f'experiments/reward_scale/{problem}/td/td.json')
    scales = td_exp._d['metaParameters']['reward_scale']

    tdrc_exp = loadExperiment(f'experiments/reward_scale/{problem}/regh_tdc/regh_tdc.json')
    betas = tdrc_exp._d['metaParameters']['reg_h']

    mat = np.zeros((len(scales), len(betas)))

    xs = []
    ys = []
    alphas = []
    for i, scale in enumerate(scales):
        td_results = loadResults(td_exp, errorfile)
        td_results = whereParameterEquals(td_results, 'reward_scale', scale)
        best_td = getBest(td_results)

        best_td_mean = np.mean(best_td.mean())
        best_td_std = np.mean(np.sqrt(best_td.stderr() * np.sqrt(best_td.runs())))

        tdrc_results = loadResults(tdrc_exp, errorfile)
        tdrc_results = whereParameterEquals(tdrc_results, 'reward_scale', scale)
        tdrc_split = splitOverParameter(tdrc_results, 'reg_h')

        for j, beta in enumerate(betas):
            results = tdrc_split[beta]
            best_tdrc = getBest(results)
            mean = np.mean(best_tdrc.mean())

            stds_away = (mean - best_td_mean) / best_td_std

            # if TDRC is worse, scale between red and blue
            if stds_away > 0:
                alpha = stds_away

            # if TDRC is better, just make it blue
            else:
                alpha = 0

            xs.append(scale)
            ys.append(beta)
            alphas.append(alpha)
            mat[i, len(betas) - j - 1] = alpha

    best = min(alphas)

    cmap = cm.get_cmap('bwr')
    norm = mpl.colors.Normalize(vmin=best, vmax=MAX)

    sc = plt.scatter(xs, ys, c=alphas, cmap=cmap, norm=norm)

    # sc = plt.imshow(mat, cmap=cmap, norm=norm, interpolation='bicubic')

    plt.hlines(1, 0, max(scales), color='grey', alpha=0.5)

    ax.set_xscale("log", basex=10)
    ax.set_yscale("log", basey=2)

    cbar = plt.colorbar(sc)
    # cbar.set_ticks([0, MAX / 2, MAX])

    # plt.show()
    # exit()

    save_path = 'experiments/reward_scale/plots'
    os.makedirs(save_path, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/betas_{error}_{problem}_{bestBy}.pdf', bbox_inches='tight', dpi=100)
