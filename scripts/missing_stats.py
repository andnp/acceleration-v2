import time
import sys
import os
sys.path.append(os.getcwd())

from src.utils.model import loadExperiment
from PyExpUtils.runner import SlurmArgs
from PyExpUtils.results.paths import listResultsPaths
from PyExpUtils.utils.generator import group
from PyExpUtils.runner.Slurm import schedule, slurmOptionsFromFile

if len(sys.argv) < 3:
    print('Please run again using')
    print('python scripts/missing_stats.py [base_path] [paths/to/descriptions]...')
    exit(0)

def generateMissing(paths):
    for i, p in enumerate(paths):
        summary_path = p + '/errors_summary.npy'
        if not os.path.exists(summary_path):
            yield i

def count(gen):
    c = 0
    for p in gen:
        c += 1

    return c

experiment_paths = sys.argv[2:]

total = 0
total_missing = 0
for path in experiment_paths:
    print(path)
    exp = loadExperiment(path)

    size = exp.permutations()

    paths = listResultsPaths(exp, 1)
    indices = generateMissing(paths)
    missing = count(indices)

    total += size
    total_missing += missing

    print(missing, size, missing / size)

print('totals:')
print(total_missing, total, total_missing / total)
