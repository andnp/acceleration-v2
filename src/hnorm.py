import numpy as np
import tarfile
import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

from RlGlue import RlGlue
from src.problems.registry import getProblem
from src.utils.arrays import fillRest
from src.utils.model import loadExperiment
from src.utils.path import up
from src.utils.random import sample

SAMPLE_EVERY=25

# get the experiment model from JSON file
exp = loadExperiment(sys.argv[2])
idx = int(sys.argv[3])
RUNS = int(sys.argv[1])

run_hnorm = []
for run in range(RUNS):
    print(run)
    np.random.seed(run)

    # get problem specific settings
    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx)
    env = problem.getEnvironment()
    rep = problem.getRepresentation()

    problem.setupIdealH()

    agent_wrapper = problem.getAgent()
    glue = RlGlue(agent_wrapper, env)

    # Run the experiment
    hnorm = []
    glue.start()
    for step in range(problem.getSteps()):
        r, o, a, t = glue.step()
        if t:
            glue.start()

        # get access to original agent
        h = agent_wrapper.agent.theta[1]
        # h = agent_wrapper.agent.getIdealH()
        norm = np.linalg.norm(h)

        hnorm.append(norm)

    run_hnorm.append(hnorm)


mean = np.mean(run_hnorm, 0)
stderr = np.std(run_hnorm, 0, ddof=1) / np.sqrt(RUNS)

# plt.plot(mean)
# plt.show()
# exit()

# save things to disk
save_context = exp.buildSaveContext(idx)
save_context.ensureExists()

np.save(save_context.resolve('hnorm_summary.npy'), np.array([ mean, stderr, RUNS ]))
