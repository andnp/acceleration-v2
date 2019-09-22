import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import plot, save, plotBest
from src.analysis.results import loadResults, whereParameterEquals, getBest, getBestEnd, find, whereParameterGreaterEq
from src.analysis.colormap import colors
from src.utils.model import loadExperiment
from src.utils.path import up
from src.utils.arrays import first

from src.utils.path import fileName, up

stepsize = sys.argv[1]
bestBy = sys.argv[2]

if bestBy == 'auc':
    metric = getBest
elif bestBy == 'end':
    metric = getBestEnd

def generatePlot(exp_paths):
    f, axes = plt.subplots(2,2)

    # get LSTD solution
    path = up(up(first(exp_paths))) + '/lstd.json'
    exp = loadExperiment(path)
    LSTD_rmsve_results = loadResults(exp, 'errors_summary.npy')
    LSTD_rmspbe_results = loadResults(exp, 'rmspbe_summary.npy')

    LSTD_rmsve = metric(LSTD_rmsve_results)
    LSTD_rmspbe = metric(LSTD_rmspbe_results)

    rmspbe_bounds = []
    rmsve_bounds = []

    bounds = plotBest(LSTD_rmspbe, axes[0, 0], color=colors['LSTD'], label='LSTD', alphaMain=0.5)
    rmspbe_bounds.append(bounds)

    bounds = plotBest(LSTD_rmspbe, axes[0, 1], color=colors['LSTD'], label='LSTD', alphaMain=0.5)
    rmspbe_bounds.append(bounds)

    bounds = plotBest(LSTD_rmsve, axes[1, 0], color=colors['LSTD'], label='LSTD', alphaMain=0.5)
    rmsve_bounds.append(bounds)

    bounds = plotBest(LSTD_rmsve, axes[1, 1], color=colors['LSTD'], label='LSTD', alphaMain=0.5)
    rmsve_bounds.append(bounds)


    # RMSPBE plots
    ax = axes[0,0]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, 'rmspbe_summary.npy')
        const, unconst = tee(results)

        const = whereParameterGreaterEq(const, 'ratio', 1)

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)

        agent = exp.agent
        if 'SmoothTDC' in agent:
            average = exp._d['metaParameters']['averageType']
            agent += '_' + average

        color = colors[agent]
        label = agent

        if not (exp.agent in ['TDadagrad', 'TDschedule', 'TD', 'TDamsgrad'] or use_ideal_h):
            bounds = plot(unconst, ax, label=label + '_unc', color=color, dashed=True, bestBy=bestBy)
            rmspbe_bounds.append(bounds)
            bounds = plot(const, ax, label=label, color=color, dashed=False, bestBy=bestBy)
            rmspbe_bounds.append(bounds)
        else:
            bounds = plot(unconst, ax, label=label, color=color, dashed=False, bestBy=bestBy)
            rmspbe_bounds.append(bounds)

        ax.set_ylabel("RMSPBE")
        ax.set_title("RMSPBE")

    ax = axes[0,1]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        rmsve = loadResults(exp, 'errors_summary.npy')
        results = loadResults(exp, 'rmspbe_summary.npy')
        const, unconst = tee(rmsve)
        const_res, unconst_res = tee(results)

        const = whereParameterGreaterEq(const, 'ratio', 1)

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)

        agent = exp.agent
        if 'SmoothTDC' in agent:
            average = exp._d['metaParameters']['averageType']
            agent += '_' + average

        color = colors[agent]
        label = agent

        if not (exp.agent in ['TDadagrad', 'TDschedule', 'TD', 'TDamsgrad'] or use_ideal_h):
            best = metric(const)
            best_unc = metric(unconst)
            best_rmspbe = find(const_res, best)
            best_rmspbe_unc = find(unconst_res, best_unc)

            bounds = plotBest(best_rmspbe, ax, label=label, color=color, dashed=False)
            rmspbe_bounds.append(bounds)

            bounds = plotBest(best_rmspbe_unc, ax, label=label + '_unc', color=color, dashed=True)
            rmspbe_bounds.append(bounds)
        else:
            best = metric(unconst)
            best_rmspbe = find(unconst_res, best)

            bounds = plotBest(best_rmspbe, ax, label=label, color=color, dashed=False)
            rmspbe_bounds.append(bounds)

        ax.set_title("RMSVE")

    # RMSVE plots
    ax = axes[1,0]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        rmsve = loadResults(exp, 'errors_summary.npy')
        rmspbe = loadResults(exp, 'rmspbe_summary.npy')
        const, unconst = tee(rmspbe)
        const_res, unconst_res = tee(rmsve)

        const = whereParameterGreaterEq(const, 'ratio', 1)

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)

        agent = exp.agent
        if 'SmoothTDC' in agent:
            average = exp._d['metaParameters']['averageType']
            agent += '_' + average

        color = colors[agent]
        label = agent

        if not (exp.agent in ['TDadagrad', 'TDschedule', 'TD', 'TDamsgrad'] or use_ideal_h):
            # best PBE using AUC
            best = metric(const)
            best_unc = metric(unconst)
            best_rmsve = find(const_res, best)
            best_rmsve_unc = find(unconst_res, best_unc)

            print('rmsve_over_rmspbe')
            print(label, best_rmsve.params)
            print(label, best_rmsve_unc.params)

            bounds = plotBest(best_rmsve, ax, label=label, color=color, dashed=False)
            rmsve_bounds.append(bounds)

            bounds = plotBest(best_rmsve_unc, ax, label=label + '_unc', color=color, dashed=True)
            rmsve_bounds.append(bounds)

        else:
            best = metric(unconst)
            best_rmsve = find(unconst_res, best)
            bounds = plotBest(best_rmsve, ax, label=label, color=color, dashed=False)
            rmsve_bounds.append(bounds)

        ax.set_ylabel("RMSVE")

    ax = axes[1,1]
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp)
        const, unconst = tee(results)

        const = whereParameterGreaterEq(const, 'ratio', 1)

        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)

        agent = exp.agent
        if 'SmoothTDC' in agent:
            average = exp._d['metaParameters']['averageType']
            agent += '_' + average

        color = colors[agent]
        label = agent

        if not (exp.agent in ['TDadagrad', 'TDschedule', 'TD', 'TDamsgrad'] or use_ideal_h):
            bounds = plot(const, ax, label=label, color=color, dashed=False, bestBy=bestBy)
            rmsve_bounds.append(bounds)

            bounds = plot(unconst, ax, label=label + '_unc', color=color, dashed=True, bestBy=bestBy)
            rmsve_bounds.append(bounds)

        else:
            bounds = plot(unconst, ax, label=label, color=color, dashed=False, bestBy=bestBy)
            rmsve_bounds.append(bounds)

    # rmspbe
    rmspbe_lower = min(map(lambda x: x[0], rmspbe_bounds)) * 0.9
    rmspbe_upper = max(map(lambda x: x[1], rmspbe_bounds)) * 1.05

    if rmspbe_lower < 0.01:
        rmspbe_lower = -0.01

    axes[0, 0].set_ylim([rmspbe_lower, rmspbe_upper])
    axes[0, 1].set_ylim([rmspbe_lower, rmspbe_upper])

    # rmsve
    rmsve_lower = min(map(lambda x: x[0], rmsve_bounds)) * 0.9
    rmsve_upper = max(map(lambda x: x[1], rmsve_bounds)) * 1.05

    if rmsve_lower < 0.01:
        rmsve_lower = -0.01

    axes[1, 0].set_ylim([rmsve_lower, rmsve_upper])
    axes[1, 1].set_ylim([rmsve_lower, rmsve_upper])

if __name__ == "__main__":
    exp_paths = sys.argv[3:]

    paths = []
    for exp_path in exp_paths:
        if 'lstd' in exp_path or 'htd' in exp_path or 'ema' in exp_path:
            continue

        if stepsize != 'constant' and stepsize not in exp_path:
            continue

        if stepsize == 'constant' and ('amsgrad' in exp_path or 'adagrad' in exp_path or 'schedule' in exp_path):
            continue

        exp = loadExperiment(exp_path)
        use_ideal_h = exp._d['metaParameters'].get('use_ideal_h', False)
        if use_ideal_h:
            continue

        paths.append(exp_path)

    generatePlot(paths)

    # plt.show()

    exp_name = fileName(up(exp.getExperimentName()))
    save_path = f'experiments/stepsizes/plots/2x2/{bestBy}'
    os.makedirs(save_path, exist_ok=True)

    fig = plt.gcf()
    fig.set_size_inches((13, 12), forward=False)
    plt.savefig(f'{save_path}/{exp_name}_square_{stepsize}.png', dpi=125)
