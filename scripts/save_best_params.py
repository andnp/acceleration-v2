import os
import sys
import json
import numpy as np
sys.path.append(os.getcwd())

from src.analysis.results import loadResults, getBest
from src.utils.model import loadExperiment
from src.utils.path import up, fileName

exp_paths = sys.argv[1:]

for exp_path in exp_paths:
    exp = loadExperiment(exp_path)
    results = loadResults(exp)
    best = getBest(results)
    print('---------------------')
    print('agent:', exp.agent)
    print(best.params)

    f = fileName(exp_path)
    new_path = up(exp_path) + '/best/' + f

    d = exp._d
    d['metaParameters'] = best.params
    os.makedirs(up(new_path), exist_ok=True)
    with open(new_path, 'w') as f:
        json.dump(d, f, indent=4)
