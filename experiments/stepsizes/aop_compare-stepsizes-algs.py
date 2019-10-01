import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import lineplot
from src.analysis.results import loadResults, whereParameterGreaterEq, getBest, find
from src.analysis.colormap import stepsize_colors as colors
from src.utils.model import loadExperiment

from src.utils.arrays import first
from src.utils.path import fileName, up

error = 'rmspbe'

# name = 'policy'
# problems = ['SmallChainTabular5050', 'SmallChainTabular4060', 'Baird']

# name = 'features'
# problems = ['SmallChainTabular5050', 'SmallChainInverted5050', 'SmallChainDependent5050' 'Boyan']

name = 'all'
problems = ['SmallChainTabular5050', 'SmallChainTabular4060', 'SmallChainInverted5050', 'SmallChainInverted4060', 'SmallChainDependent5050', 'SmallChainDependent4060', 'Boyan', 'Baird']

algorithms = ['gtd2', 'tdc']
stepsizes = ['constant', 'schedule']

if error == 'rmsve':
    errorfile = 'errors_summary.npy'
elif error == 'rmspbe':
    errorfile = 'rmspbe_summary.npy'

if __name__ == "__main__":
    f, ax = plt.subplots(1, 1)

    all_steps = []
    for problem in problems:
        path = f'experiments/stepsizes/{problem}/lstd.json'
        lstd_exp = loadExperiment(path)
        results = loadResults(lstd_exp, errorfile)
        best = getBest(results)

        curve = best.mean()
        steps = len(curve)
        all_steps.append(steps)

    max_steps = min(all_steps)
    curves = np.zeros((len(stepsizes), len(algorithms), len(problems), max_steps))

    for k, problem in enumerate(problems):
        for i, ss in enumerate(stepsizes):
            for j, alg in enumerate(algorithms):
                print(problem, alg, ss)
                if ss == 'constant':
                    exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/{alg}.json')
                else:
                    exp_paths = glob.glob(f'experiments/stepsizes/{problem}/{alg}/{alg}{ss}.json')

                exp = loadExperiment(exp_paths[0])

                results = loadResults(exp, errorfile)
                results = whereParameterGreaterEq(results, 'ratio', 1)
                best = getBest(results)

                curve = best.mean()[:max_steps]

                curves[i, j, k, :] = curve / curve[0]


    aop = np.mean(curves, axis=2)
    sop = np.std(curves, axis=2) / np.sqrt(len(problems))
    for i, ss in enumerate(stepsizes):
        for j, alg in enumerate(algorithms):
            curve = aop[i, j]
            stderr = sop[i, j]
            stepsize = ss
            if ss == 'constant':
                stepsize = ''

            agent = alg.upper() + stepsize
            color = colors[agent]
            label = agent

            dashed = False
            if 'TDC' in label:
                dashed = True

            lineplot(ax, curve, stderr, color=color, label=label, dashed=dashed)



    plt.show()
    exit()

    save_path = 'experiments/stepsizes/plots'
    os.makedirs(save_path, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/{name}_aop-compare-stepsizes-algs_{error}.png', dpi=100)
