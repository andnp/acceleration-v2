import subprocess
import multiprocessing
from PyExpUtils.runner import Args, parallel
from PyExpUtils.results.indices import listMissingResults, listIndices
from src.utils.model import loadExperiment


args = Args.fromCommandLine()

for path in args.experiment_paths:
    exp = loadExperiment(path)

    # get all of the indices corresponding to missing results
    indices = listIndices(exp, args.runs) if args.retry else listMissingResults(exp, args.runs)

    # build the parallel command
    parallel_cmd = parallel.buildParallel({
        'executable': args.executable + ' ' + path,
        'tasks': indices,
        'cores': multiprocessing.cpu_count(),
    })

    parallel_cmd = parallel_cmd.insist()

    if len(parallel_cmd.split(' ::: ')[1]) == 0:
        continue

    subprocess.run(parallel_cmd, stdout=subprocess.PIPE, shell=True)
