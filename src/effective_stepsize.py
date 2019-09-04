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
from src.utils.stats import exponentialSmoothing

SAMPLE_EVERY = 100

# get the experiment model from JSON file
exp = loadExperiment(sys.argv[2])
idx = int(sys.argv[3])
RUNS = int(sys.argv[1])

run_stepsizes = []
for run in range(RUNS):
    print(run)
    np.random.seed(run)
    random.seed(a=run)

    # get problem specific settings
    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx)
    env = problem.getEnvironment()
    rep = problem.getRepresentation()

    # sort of a hack, but this only needs to happen the first time
    if run == 0:
        experience_generator = problem.sampleExperiences()

    # set up the MDP for computing h*
    problem.setupIdealH()

    agent_wrapper = problem.getAgent()
    glue = RlGlue(agent_wrapper, env)

    # Run the experiment
    stepsizes = []
    glue.start()
    broke = False
    for step in range(problem.getSteps()):
        r, o, a, t = glue.step()
        if t:
            glue.start()

        if step % SAMPLE_EVERY == 0:
            experiences = experience_generator.sample(1000)
            ss_w, ss_h = problem.agent.effectiveStepsize(experiences)

            stepsizes.append([np.mean([ss_w, ss_h]), ss_w, ss_h])


    run_stepsizes.append(stepsizes)

print(np.array(run_stepsizes).shape)

mean = exponentialSmoothing(np.mean(run_stepsizes, 0))
stderr = np.std(run_stepsizes, 0, ddof=1) / np.sqrt(RUNS)

for m, label in zip(mean.T, ['mean', 'w', 'h']):
    plt.plot(m, label=label)
plt.legend()
plt.show()
exit()

# save things to disk
save_context = exp.buildSaveContext(idx)
save_context.ensureExists()

np.save(save_context.resolve('stepsize_summary.npy'), np.array([ mean, stderr, RUNS ]))
