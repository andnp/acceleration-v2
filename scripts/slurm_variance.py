import time
import sys
import os
from src.utils.model import loadExperiment
from PyExpUtils.runner import SlurmArgs
from PyExpUtils.results.paths import listResultsPaths
from PyExpUtils.utils.generator import group
from PyExpUtils.runner.Slurm import schedule, slurmOptionsFromFile

if len(sys.argv) < 4:
    print('Please run again using')
    print('python -m scripts.scriptName [path/to/slurm-def] [base_path] [runs] [paths/to/descriptions]...')
    exit(0)

args = SlurmArgs.SlurmArgsModel({
    'experiment_paths': sys.argv[4:],
    'base_path': sys.argv[2],
    'runs': 1,
    'slurm_path': sys.argv[1],
    'executable': "python src/update_variance " + sys.argv[3],
})

def generateMissing(paths):
    for i, p in enumerate(paths):
        summary_path = '/'.join(p.split('/')[:-1]) + '/variance_summary.npy'
        if not os.path.exists(summary_path):
            yield i

for path in args.experiment_paths:
    exp = loadExperiment(path)
    slurm = slurmOptionsFromFile(args.slurm_path)

    paths = listResultsPaths(exp, args.runs)
    indices = generateMissing(paths)

    groupSize = slurm.tasks * slurm.tasksPerNode

    for g in group(indices, groupSize):
        l = list(g)
        print("scheduling:", path, l)
        slurm.tasks = min([slurm.tasks, len(l)])
        schedule(slurm, args.executable + ' ' + path, l)
        time.sleep(2)
