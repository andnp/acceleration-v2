import numpy as np
import random
import sys
import os
sys.path.append(os.getcwd())

from src.problems.registry import getProblem
from src.utils.model import loadExperiment
from src.utils.SampleGenerator import SampleGenerator
from src.utils.Collector import Collector

# get the experiment model from JSON file
exp = loadExperiment(sys.argv[2])
idx = int(sys.argv[3])
RUNS = int(sys.argv[1])

EVERY = exp.subsample

if exp.agent == 'LSTD':
    exit()

collector = Collector()

num_params = exp.permutations()
for run in range(RUNS):
    np.random.seed(run)
    random.seed(a=run)

    # get problem specific settings
    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx + run * num_params)
    env = problem.getEnvironment()
    rep = problem.getRepresentation()

    # determine effective number of samples
    # we need 300 at most
    steps = problem.exp.steps
    samples = steps // EVERY
    if samples > 300:
        EVERY = int(steps // 300)

    agent = problem.agent

    if run % 50 == 0:
        generator = SampleGenerator(problem)
        generator.generate(num=1e6)

    # Run the experiment
    broke = False
    for step in range(steps):
        agent.batch_update(generator)

        if step % EVERY != 0:
            continue

        # collect error from problem definition
        rmsve, rmspbe = problem.evaluateStep(None)

        collector.collect('errors', rmsve)
        collector.collect('rmspbe', rmspbe)

        # if we've diverged, just go ahead and give up
        # saves some computation and these runs are useless to me anyways
        if np.isnan(rmsve) or np.isinf(rmsve):
            collector.fillRest(np.nan, int(problem.getSteps() / EVERY))
            broke = True
            break

    collector.reset()
    if broke:
        break


# get stats over runs for each collected variable
error_data = collector.getStats('errors')
rmspbe_data = collector.getStats('rmspbe')

# local plotting (for testing)

# import matplotlib.pyplot as plt
# from src.utils.plotting import plot
# fig, (ax1, ax2) = plt.subplots(1, 2)

# plot(ax1, error_data)
# ax1.set_title('RMSVE')
# plot(ax2, rmspbe_data)
# ax2.set_title('RMSPBE')

# plt.show()
# exit()

# save things to disk
save_context = exp.buildSaveContext(idx)
save_context.ensureExists()

np.save(save_context.resolve('errors_summary.npy'), np.array(error_data))
np.save(save_context.resolve('rmspbe_summary.npy'), np.array(rmspbe_data))
