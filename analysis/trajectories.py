import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
sys.path.append(os.getcwd())

from src.analysis.learning_curve import plot, save
from src.analysis.results import loadResults, whereParameterEquals
from src.utils.model import loadExperiment

from src.utils.path import fileName

def generatePlot(exp_paths):
    ax = plt.gca()
    # ax.semilogx()
    weights = {}
    for exp_path in exp_paths:
        exp = loadExperiment(exp_path)
        results = loadResults(exp, "weight_trajectory.npy")

        label = fileName(exp_path).replace('.json', '')
        w = next(results)._lazyLoad()
        print(w.shape)
        weights[label] = w

    keys = weights.keys()
    for k in keys:
        if k != "td":
            w1_all=weights[k]; w2_all=weights["td"]
            sim = []
            for r in range(w1_all.shape[0]):
                w1=w1_all[r]; w2=w2_all[r]
                cos_sim = []
                for t in range(w1.shape[0]):
                    cos_sim.append(
                        np.dot(w1[t],w2[t]) / (np.linalg.norm(w1[t])*np.linalg.norm(w2[t]))
                    )
                sim.append(cos_sim)
            cos_sim = np.mean(sim, 0)
            plt.plot(cos_sim, label="{}/{} cosine similarity".format(k,'td'))
    plt.legend()
    plt.show()
    plt.clf()

    for k in keys:
        if k != "td":
            w1_all=weights[k]; w2_all=weights["td"]
            sim = []
            for r in range(w1_all.shape[0]):
                w1=w1_all[r]; w2=w2_all[r]
                rank_corr = []
                for t in range(w1.shape[0]):
                    rank_corr.append(
                        stats.kendalltau(w1[t],w2[t])[0]
                    )
                sim.append(rank_corr)
            rank_corr = np.mean(sim, 0)
            plt.plot(rank_corr, label="{}/{} rank similarity".format(k,'td'))
    plt.legend()
    plt.show()
    plt.clf()

    for k in keys:
        if k != "td":
            w1_all=weights[k]; w2_all=weights["td"]
            sim = []
            for r in range(w1_all.shape[0]):
                w1=w1_all[r]; w2=w2_all[r]
                l2 = []
                for t in range(w1.shape[0]):
                    l2.append(
                        np.linalg.norm(w1[t]-w2[t])
                    )
                sim.append(l2)
            l2 = np.mean(sim,0)
            plt.plot(l2, label="{}/{} rank similarity".format(k,'td'))
    plt.legend()
    plt.show()
    plt.clf()

if __name__ == "__main__":
    exp_paths = sys.argv[1:]
    tmp = loadExperiment(exp_paths[0])

    generatePlot(exp_paths)
