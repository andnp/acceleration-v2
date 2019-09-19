import glob
import subprocess

problems = glob.glob('experiments/stepsizes/*')

def last(path):
    parts = path.split('/')
    return parts[len(parts) - 1]

problems = [p for p in problems if '.' not in last(p) and last(p) != 'plots']

for problem in problems:
    experiments = problem + '/**/*.json'

    subprocess.run(f'python experiments/stepsizes/slice_combs.py {experiments}', stdout=subprocess.PIPE, shell=True)
