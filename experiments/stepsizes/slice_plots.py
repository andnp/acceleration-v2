import glob
import subprocess

problems = glob.glob('experiments/stepsizes/*')
stepsizes = ['amsgrad', 'adagrad', 'schedule', 'constant']
metrics = ['end', 'auc']

def last(path):
    parts = path.split('/')
    return parts[len(parts) - 1]

problems = [p for p in problems if '.' not in last(p) and last(p) != 'plots']

for bestBy in metrics:
    for ss in stepsizes:
        for problem in problems:
            experiments = problem + '/**/*.json'

            subprocess.run(f'python experiments/stepsizes/slice_combs.py {ss} {bestBy} {experiments}', stdout=subprocess.PIPE, shell=True)
