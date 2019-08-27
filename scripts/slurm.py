import time
from PyExpUtils.runner import SlurmArgs
from src.utils.model import loadExperiment
from PyExpUtils.results.indices import listMissingResults, listIndices
from PyExpUtils.utils.generator import group
from PyExpUtils.runner.Slurm import schedule, slurmOptionsFromFile

args = SlurmArgs.fromCommandLine()
for path in args.experiment_paths:
    exp = loadExperiment(path)
    slurm = slurmOptionsFromFile(args.slurm_path)

    # get all of the indices corresponding to missing results
    indices = listIndices(exp, args.runs) if args.retry else listMissingResults(exp, args.runs)

    groupSize = slurm.tasks * slurm.tasksPerNode

    for g in group(indices, groupSize):
        l = list(g)
        print("scheduling:", path, l)
        schedule(slurm, args.executable + ' ' + path, l)
        time.sleep(2)
