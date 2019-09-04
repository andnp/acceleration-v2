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

run_variances = []
for run in range(RUNS):
    print(run)
    np.random.seed(run)

    # get problem specific settings
    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx)
    env = problem.getEnvironment()
    rep = problem.getRepresentation()

    # sort of a hack, but this only needs to happen the first time
    if run == 0:
        experience_generator = problem.sampleExperiences()

    agent_wrapper = problem.getAgent()
    glue = RlGlue(agent_wrapper, env)

    # Run the experiment
    variances = []
    glue.start()
    for step in range(problem.getSteps()):
        r, o, a, t = glue.step()
        if t:
            glue.start()

        if step % SAMPLE_EVERY == 0:
            experiences = experience_generator.sample(1000)
            var_w, var_h = problem.agent.computeVarianceOfUpdates(experiences)

            variances.append([np.mean([var_w, var_h]), var_w, var_h])

    run_variances.append(variances)


mean = np.mean(run_variances, 0)
stderr = np.std(run_variances, 0, ddof=1) / np.sqrt(RUNS)

# plt.plot(mean)
# plt.show()
# exit()

# save things to disk
save_context = exp.buildSaveContext(idx)
save_context.ensureExists()

np.save(save_context.resolve('variance_summary.npy'), np.array([ mean, stderr, RUNS ]))
