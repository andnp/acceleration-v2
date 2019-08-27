import os
import sys
import numpy as np
sys.path.append(os.getcwd())

from src.analysis.results import loadResults, getBest, whereParameterEquals
from src.utils.model import loadExperiment

exp_paths = sys.argv[1:]

for exp_path in exp_paths:
    exp = loadExperiment(exp_path)
    results = loadResults(exp)
    results = whereParameterEquals(results, 'lambda', 0.1)
    best = getBest(results)
    print('---------------------')
    print('agent:', exp.agent)
    print(best.params)

    # mean over runs
    mean = best.mean()
    steps = len(mean)
    print('mean:', np.mean(mean))
    print('end:', np.mean(mean[-int(.1 * steps):]))
