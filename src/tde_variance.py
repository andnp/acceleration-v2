import numpy as np
import tarfile
import sys
import os
sys.path.append(os.getcwd())

from multiprocessing.pool import Pool
from functools import partial
import matplotlib.pyplot as plt

from RlGlue import RlGlue
from src.problems.registry import getProblem
from src.utils.arrays import fillRest
from src.utils.model import loadExperiment
from src.utils.path import up

SAMPLE_EVERY = 25

def startRun(exp, idx, run):
    print(run)
    np.random.seed(run)

    # get problem specific settings
    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx)
    env = problem.getEnvironment()
    rep = problem.getRepresentation()

    experience_generator = problem.sampleExperiences()
    agent = problem.getAgent()
    glue = RlGlue(agent, env)

    # Run the experiment
    variances = []
    glue.start()
    for step in range(problem.getSteps()):
        r, o, a, t = glue.step()
        if t:
            glue.start()

        if step % SAMPLE_EVERY != 0:
            continue

        experiences = experience_generator.sample(1000)
        var = problem.agent.computeVarianceOfTDE(experiences)

        variances.append(var)

    return variances

if __name__ == "__main__":
    # get the experiment model from JSON file
    exp = loadExperiment(sys.argv[2])
    idx = int(sys.argv[3])
    RUNS = int(sys.argv[1])

    pool = Pool(5)

    if not exp.agent.startswith('td'):
        raise Exception("Can only run with a TD agent for now")

    run_variances = pool.map(partial(startRun, exp, idx), range(RUNS))

    mean = np.mean(run_variances, 0)
    stderr = np.std(run_variances, 0, ddof=1) / np.sqrt(RUNS)

    plt.plot(mean)
    plt.show()
    exit()

    # save things to disk
    save_context = exp.buildSaveContext(idx)
    save_context.ensureExists()

    path = up(save_context.resolve())
    np.save(path + '/tde_variance_summary.npy', np.array([ mean, stderr, RUNS ]))
