import numpy as np
import random
import tarfile
import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

from RlGlue import RlGlue
from src.problems.registry import getProblem
from src.utils.arrays import fillRest
from src.utils.model import loadExperiment

# get the experiment model from JSON file
exp = loadExperiment(sys.argv[2])
idx = int(sys.argv[3])
RUNS = int(sys.argv[1])

run_errors = []
for run in range(RUNS):
    np.random.seed(run)
    random.seed(a=run)

    # get problem specific settings
    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx)
    env = problem.getEnvironment()
    rep = problem.getRepresentation()

    # set up the MDP for computing h*
    problem.setupIdealH()

    agent_wrapper = problem.getAgent()
    glue = RlGlue(agent_wrapper, env)

    # Run the experiment
    errors = []
    glue.start()
    broke = False
    for step in range(problem.getSteps()):
        r, o, a, t = glue.step()
        if t:
            glue.start()

        e = problem.evaluateStep({
            'step': step,
            'reward': r,
        })

        # if we've diverged, just go ahead and give up
        # saves some computation and these runs are useless to me anyways
        if np.isnan(e) or np.isinf(e):
            fillRest(errors, np.nan, problem.getSteps())
            broke = True
            break

        errors.append(e)

    run_errors.append(errors)
    if broke:
        break


mean = np.mean(run_errors, 0)
stderr = np.std(run_errors, 0, ddof=1) / np.sqrt(RUNS)

# plt.plot(mean)
# plt.show()
# exit()

# save things to disk
save_context = exp.buildSaveContext(idx)
save_context.ensureExists()

np.save(save_context.resolve('errors_summary.npy'), np.array([ mean, stderr, RUNS ]))
