import os
import sys
import subprocess
import multiprocessing
from functools import partial
from multiprocessing.pool import Pool
from PyExpUtils.runner import Args, parallel
from PyExpUtils.results.paths import listResultsPaths
from src.utils.model import loadExperiment
from src.utils.arrays import first

def generateMissing(paths):
    for i, p in enumerate(paths):
        summary_path = p + '/variance_summary.npy'
        if not os.path.exists(summary_path):
            yield i

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Please run again using')
        print('python -m scripts.scriptName [runs] [base_path] [paths/to/descriptions]...')
        exit(0)

    args = Args.ArgsModel({
        'experiment_paths': sys.argv[3:],
        'base_path': sys.argv[2],
        'runs': 1,
        'executable': "python src/update_variance.py " + sys.argv[1],
    })

    pool = Pool()

    cmds = []
    for path in args.experiment_paths:
        exp = loadExperiment(path)

        paths = listResultsPaths(exp, args.runs)
        indices = generateMissing(paths)

        exe = args.executable + ' ' + path + ' ' + str(first(indices))
        cmds.append(exe)

    pool.map(partial(subprocess.run, shell=True, stdout=subprocess.PIPE), cmds)
