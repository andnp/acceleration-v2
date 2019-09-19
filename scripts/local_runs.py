import sys
import os
sys.path.append(os.getcwd())

import subprocess
import multiprocessing
from PyExpUtils.runner import Args, parallel
from PyExpUtils.results.indices import listMissingResults, listIndices
from src.utils.model import loadExperiment

if len(sys.argv) < 4:
    print('Please run again using')
    print('python scripts/local_runs.py [runs] [base/path/to/results] [paths/to/descriptions]...')
    exit(0)

runs = sys.argv[1]
args = Args.ArgsModel({
    'experiment_paths': sys.argv[3:],
    'base_path': sys.argv[2],
    'runs': 1,
    'executable': "python src/runs.py",
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

    try:
        parallel_cmd = parallel_cmd.insist()
    except:
        continue

    if len(parallel_cmd.split(' ::: ')[1]) == 0:
        continue

    subprocess.run(parallel_cmd, stdout=subprocess.PIPE, shell=True)
    print('finished: ', path)
