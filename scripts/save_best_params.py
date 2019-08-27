import os
import sys
import json
import numpy as np
sys.path.append(os.getcwd())

from src.analysis.results import loadResults, getBest
from src.utils.model import loadExperiment

exp_paths = sys.argv[1:]

for exp_path in exp_paths:
    exp = loadExperiment(exp_path)
    results = loadResults(exp)
    best = getBest(results)
    print('---------------------')
    print('agent:', exp.agent)
    print(best.params)

    new_path = exp_path.replace('/sweeps/', '/best/')
    d = exp._d
    d['metaParameters'] = best.params
    os.makedirs('/'.join(new_path.split('/')[:-1]), exist_ok=True)
    with open(new_path, 'w') as f:
        json.dump(d, f, indent=4)
