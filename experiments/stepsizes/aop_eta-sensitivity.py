import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.sensitivity_curve import getSensitivityData, sensitivityCurve
from src.analysis.learning_curve import lineplot
from src.analysis.results import loadResults, whereParameterGreaterEq, whereParameterEquals, getBest, find
from src.analysis.colormap import colors
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

error = 'rmspbe'

# name = 'on-policy'
# problems = ['SmallChainTabular5050', 'SmallChainInverted5050', 'SmallChainDependent5050', 'Boyan']

# name = 'off-policy'
# problems = ['SmallChainTabular4060', 'SmallChainInverted4060', 'SmallChainDependent4060']

name = 'all'
problems = ['SmallChainTabular5050', 'SmallChainTabular4060', 'SmallChainInverted5050', 'SmallChainInverted4060', 'SmallChainDependent5050', 'SmallChainDependent4060', 'Boyan']

algorithms = ['gtd2', 'tdc', 'regh_tdc', 'htd']
stepsizes = ['constant', 'adagrad', 'schedule']

on_policy_problems = ['SmallChainTabular5050', 'SmallChainInverted5050', 'SmallChainDependent5050', 'Boyan']

if error == 'rmsve':
    errorfile = 'errors_summary.npy'
elif error == 'rmspbe':
    errorfile = 'rmspbe_summary.npy'

if __name__ == "__main__":
    f, ax = plt.subplots(1, 1)

    exp = loadExperiment(f'experiments/stepsizes/{problems[0]}/{algorithms[0]}/{algorithms[0]}.json')
    params = len(exp._d['metaParameters']['ratio'])

    curves = np.zeros((len(stepsizes), len(algorithms) + 1, len(problems), params, 2))

    total_runs = 0
    for k, problem in enumerate(problems):
        for i, ss in enumerate(stepsizes):
            td_idx = len(algorithms)

            if ss == 'constant':
                td_path = f'experiments/stepsizes/{problem}/td/td.json'
            else:
                td_path = f'experiments/stepsizes/{problem}/td/td{ss}.json'

            exp = loadExperiment(td_path)
            results = loadResults(exp, errorfile)
            best = getBest(results)
            curve = best.mean()
            curves[i, td_idx, k, :, 0] = np.mean(curve) / curve[0]
            curves[i, td_idx, k, :, 1] = np.mean(best.stderr()) * np.sqrt(best.runs())

            for j, alg in enumerate(algorithms):
                print(problem, alg, ss)

                if alg == 'htd' and problem in on_policy_problems:
                    curves[i, j, k] = curves[i, td_idx, k]
                    continue


                if ss == 'constant':
                    exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/{alg}.json')
                else:
                    exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/{alg}{ss}.json')

                exp = loadExperiment(exp_paths[0])

                results = loadResults(exp, errorfile)
                results = whereParameterEquals(results, 'reg_h', 0.8)
                lc, results = tee(results)
                best = getBest(lc)

                x, y, e = getSensitivityData(results, 'ratio', reducer='best', bestBy='auc')

                curve = best.mean()

                curves[i, j, k, :, 0] = np.array(y) / curve[0]
                curves[i, j, k, :, 1] = np.array(e) * np.sqrt(best.runs())

                if j == 0:
                    total_runs += best.runs()

    aop = np.mean(np.mean(curves[:, :, :, :, 0], axis=2), axis=0)
    sop = np.sum(np.sum(curves[:, :, :, :, 1], axis=2), axis=0) / np.sqrt(total_runs)
    for j, alg in enumerate(algorithms + ['td']):
        curve = aop[j]
        stderr = sop[j]
        agent = alg.upper()
        color = colors[agent]
        label = agent

        alphaMain = 1
        if alg == 'td':
            alphaMain = 0.4

        sensitivityCurve(ax, x, curve, stderr, color=color, alphaMain=alphaMain, label=label)

    ax.set_ylim([0.15, 0.55])
    ax.set_xscale("log", basex=2)
    ax.legend()

    plt.show()
    exit()

    save_path = 'experiments/stepsizes/plots'
    os.makedirs(save_path, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/{name}_aop-eta-sensitivity_{error}.png', dpi=100)
