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
from src.utils.Collector import Collector

def weightedNorm(X, d):
    return np.sqrt(X.T.dot(np.diag(d)).dot(X))

# get the experiment model from JSON file
exp = loadExperiment(sys.argv[2])
idx = int(sys.argv[3])
RUNS = int(sys.argv[1])

collector = Collector()

for run in range(RUNS):
    np.random.seed(run)
    random.seed(a=run)

    # get problem specific settings
    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx)
    env = problem.getEnvironment()
    rep = problem.getRepresentation()

    is_using_ideal_h = problem.metaParameters.get('use_ideal_h', False)

    # compute the optimal set of weights
    X = problem.all_observables
    dB = np.diag(problem.db)
    w_star = np.linalg.pinv(np.dot(X.T.dot(dB), X)).dot(X.T).dot(dB).dot(problem.v_star)
    residuals = weightedNorm(np.dot(X, w_star) - problem.v_star, problem.db)

    collector.collect('residuals', residuals)

    # set up the MDP for computing h*
    if is_using_ideal_h:
        problem.setupIdealH()

    agent_wrapper = problem.getAgent()
    glue = RlGlue(agent_wrapper, env)

    # Run the experiment
    glue.start()
    broke = False
    for step in range(problem.getSteps()):
        r, o, a, t = glue.step()
        if t:
            glue.start()

        # collect error from problem definition
        rmsve, rmspbe = problem.evaluateStep({
            'step': step,
            'reward': r,
        })

        collector.collect('errors', rmsve)
        collector.collect('rmspbe', rmspbe)

    # add the run data to the global pool
    collector.reset()

err_data = collector.getStats('errors')
rmspbe_data = collector.getStats('rmspbe')
res_data = collector.getStats('residuals')
# local plotting (for testing)
fig, (ax1, ax2) = plt.subplots(1, 2)
# print(res_data)
# ax1.plot(err_data[0])
# ax2.plot(rmspbe_data[0])

# plt.show()
# exit()

# save things to disk
save_context = exp.buildSaveContext(idx)
save_context.ensureExists()

np.save(save_context.resolve('errors_summary.npy'), np.array(err_data))
np.save(save_context.resolve('rmspbe_summary.npy'), np.array(rmspbe_data))
np.save(save_context.resolve('residuals_summary.npy'), np.array(res_data))
