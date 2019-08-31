import sys
import os
sys.path.append(os.getcwd())

import subprocess
import multiprocessing
from PyExpUtils.runner import Args, parallel
from PyExpUtils.results.indices import listMissingResults, listIndices
from src.utils.model import loadExperiment

if len(sys.argv) < 5:
    print('Please run again using')
    print('python -m scripts.scriptName [path/to/executable] [runs] [base_path] [paths/to/descriptions]...')
    exit(0)

runs = sys.argv[2]
args = Args.ArgsModel({
    'experiment_paths': sys.argv[4:],
    'base_path': sys.argv[3],
    'runs': 1,
    'executable': sys.argv[1],
})

for path in args.experiment_paths:
    exp = loadExperiment(path)

    # get all of the indices corresponding to missing results
    indices = listIndices(exp, args.runs) if args.retry else listMissingResults(exp, args.runs)

    # build the parallel command
    parallel_cmd = parallel.buildParallel({
        'executable': args.executable + ' ' + runs + ' ' + path,
        'tasks': indices,
        'cores': multiprocessing.cpu_count(),
    })

    parallel_cmd = parallel_cmd.insist()

    if len(parallel_cmd.split(' ::: ')[1]) == 0:
        continue

    subprocess.run(parallel_cmd, stdout=subprocess.PIPE, shell=True)
