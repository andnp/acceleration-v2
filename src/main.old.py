import numpy as np
import tarfile
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.problems.registry import getProblem
from src.utils.arrays import fillRest
from src.utils.model import loadExperiment

# get the experiment model from JSON file
exp = loadExperiment()
idx = int(sys.argv[2])

# which run number is this
run = exp.getRun(idx)

np.random.seed(run)

# get problem specific settings
Problem = getProblem(exp.problem)
problem = Problem(exp, idx)
env = problem.getEnvironment()
rep = problem.getRepresentation()

agent = problem.getAgent()
glue = RlGlue(agent, env)

# Run the experiment
errors = []
glue.start()
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
        break

    errors.append(e)

# save things to disk
save_context = exp.buildSaveContext(idx, base='results')
save_context.ensureExists()

np.save(save_context.resolve('errors.npy'), np.array(errors))
