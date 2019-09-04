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

def cosineSim(x, y, X, w):
    a = X.dot(x)
    b = X.dot(y)
    norm_a = np.sqrt(a.T.dot(w).dot(a))
    norm_b = np.sqrt(b.T.dot(w).dot(b))

    denom = (norm_a * norm_b)
    if denom == 0:
        return 1

    return a.T.dot(w).dot(b) / denom

def diff(a, b, X, w):
    d = X.dot(a - b)
    return np.sqrt(d.T.dot(w).dot(d))

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

        # collect the norm of h
        if is_using_ideal_h:
            h = agent_wrapper.agent.getIdealH()
        else:
            h = agent_wrapper.agent.theta[1]

        w = agent_wrapper.agent.theta[0]

        h_star = agent_wrapper.agent.getIdealH()

        hsim = cosineSim(h, h_star, X, dB)
        wsim = cosineSim(w, w_star, X, dB)

        collector.collect('hsim', hsim)
        collector.collect('wsim', wsim)

        hdiff = diff(h, h_star, X, dB)
        wdiff = diff(w, w_star, X, dB)

        collector.collect('hdiff', hdiff)
        collector.collect('wdiff', wdiff)

    print(agent_wrapper.agent.theta[0])


    # add the run data to the global pool
    collector.reset()

hsim_data = collector.getStats('hsim')
wsim_data = collector.getStats('wsim')
hdiff_data = collector.getStats('hdiff')
wdiff_data = collector.getStats('wdiff')

# local plotting (for testing)
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(wsim_data[0], label='w')
ax1.plot(hsim_data[0], label='h')

ax2.plot(wdiff_data[0], label='w')
ax2.plot(hdiff_data[0], label='h')


plt.legend()

plt.show()
exit()

# save things to disk
save_context = exp.buildSaveContext(idx)
save_context.ensureExists()

np.save(save_context.resolve('hsim_summary.npy'), np.array(hsim_data))
np.save(save_context.resolve('wsim_summary.npy'), np.array(wsim_data))
