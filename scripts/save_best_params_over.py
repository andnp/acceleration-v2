import os
import sys
import json
import numpy as np
from functools import partial
from multiprocessing.pool import Pool
sys.path.append(os.getcwd())

from src.analysis.results import loadResults, getBestOverParameter
from src.utils.model import loadExperiment
from src.utils.path import up

def process(param, exp_path):
    exp = loadExperiment(exp_path)
    results = loadResults(exp)
    best_dict = getBestOverParameter(results, param)
    for key in best_dict:
        best = best_dict[key]
        print('---------------------')
        print('agent:', exp.agent)
        print(best.params)

        new_path = exp_path.replace('/sweeps/', f'/best/{param}-{key}/')
        d = exp._d
        d['metaParameters'] = best.params
        os.makedirs(up(new_path), exist_ok=True)
        with open(new_path, 'w') as f:
            json.dump(d, f, indent=4)

if __name__ == "__main__":
    pool = Pool()
    param = sys.argv[1]
    exp_paths = sys.argv[2:]
    pool.map(partial(process, param), exp_paths)
