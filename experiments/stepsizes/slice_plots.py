import glob
import subprocess
from multiprocessing.pool import Pool
from functools import partial

def runSlice(ss, bestBy, problem):
    experiments = problem + '/**/*.json'
    print(problem, ss, bestBy)
    subprocess.run(f'python experiments/stepsizes/slice_combs.py {ss} {bestBy} {experiments}', stdout=subprocess.PIPE, shell=True)

if __name__ == "__main__":
    pool = Pool()

    problems = glob.glob('experiments/stepsizes/*')
    stepsizes = ['amsgrad', 'adagrad', 'schedule', 'constant']
    metrics = ['end', 'auc']

    def last(path):
        parts = path.split('/')
        return parts[len(parts) - 1]

    problems = [p for p in problems if '.' not in last(p) and last(p) != 'plots']

    for bestBy in metrics:
        for ss in stepsizes:
            pool.map(partial(runSlice, ss, bestBy), problems)
