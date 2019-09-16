import numpy as np
import random
import tarfile
import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
from src.utils.plotting import plot

from RlGlue import RlGlue
from src.problems.registry import getProblem
from src.utils.arrays import fillRest
from src.utils.model import loadExperiment
from src.utils.Collector import Collector

EVERY = 10

# get the experiment model from JSON file
exp = loadExperiment(sys.argv[2])
RUNS = int(sys.argv[1])

collector = Collector()
for run in range(RUNS):
    np.random.seed(run)
    random.seed(a=run)

    # get problem specific settings
    Problem = getProblem(exp.problem)
    problem = Problem(exp, 0)
    env = problem.getEnvironment()
    rep = problem.getRepresentation()

    agent = problem.agent

    A = problem.A
    b = problem.b

    w = np.linalg.pinv(A).dot(b)
    agent.theta[0] = w

    rmsve, rmspbe = problem.evaluateStep(None)

    steps = int(problem.getSteps() / EVERY)

    collector.collectAll('rmsve', [rmsve] * steps)
    collector.collectAll('rmspbe', [rmspbe] * steps)


# get stats over runs for each collected variable
error_data = collector.getStats('rmsve')
rmspbe_data = collector.getStats('rmspbe')

# local plotting (for testing)
# fig, (ax1, ax2) = plt.subplots(2)

# plot(ax1, error_data)
# ax1.set_title('RMSVE')
# plot(ax2, rmspbe_data)
# ax2.set_title('RMSPBE')

# plt.show()
# exit()

# save things to disk
save_context = exp.buildSaveContext(0)
save_context.ensureExists()

np.save(save_context.resolve('errors_summary.npy'), np.array(error_data))
np.save(save_context.resolve('rmspbe_summary.npy'), np.array(rmspbe_data))
