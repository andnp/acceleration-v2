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

EVERY = 25

def weightedNorm(X, W):
    return np.sqrt(X.T.dot(W).dot(X))

def cosineSim(x, y, w):
    norm_x = weightedNorm(x, w)
    norm_y = weightedNorm(y, w)

    denom = (norm_x * norm_y)
    if denom == 0:
        return 1

    return x.T.dot(w).dot(y) / denom

def diff(a, b, w):
    d = a - b
    return weightedNorm(d, w)

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
    X = problem.X
    dB = np.diag(problem.db)

    v_star = problem.v_star
    norm_v_star = weightedNorm(v_star, dB)

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

        if step % EVERY != 0:
            continue

        # collect the norm of h
        if is_using_ideal_h:
            h = agent_wrapper.agent.getIdealH()
        else:
            h = agent_wrapper.agent.theta[1]

        w = agent_wrapper.agent.theta[0]
        v = X.dot(w)

        h_star = agent_wrapper.agent.getIdealH()

        delta = X.dot(h)
        delta_star = X.dot(h_star)

        hsim = cosineSim(delta, delta_star, dB)
        wsim = cosineSim(v, v_star, dB)

        collector.collect('delta_sim', hsim)
        collector.collect('v_sim', wsim)

        norm_delta_star = weightedNorm(delta_star, dB)
        hdiff = diff(delta, delta_star, dB) / norm_delta_star
        wdiff = diff(delta, delta_star, dB)

        collector.collect('delta_diff', hdiff)
        collector.collect('v_diff', wdiff)

        collector.collect('norm_delta_star', norm_delta_star)
        collector.collect('norm_delta', weightedNorm(delta, dB))

    # add the run data to the global pool
    collector.reset()

hsim_data = collector.getStats('delta_sim')
wsim_data = collector.getStats('v_sim')
hdiff_data = collector.getStats('delta_diff')
wdiff_data = collector.getStats('v_diff')
nds_data = collector.getStats('norm_delta_star')
nd_data = collector.getStats('norm_delta')

# local plotting (for testing)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.plot(wsim_data[0], label='w')
ax1.plot(hsim_data[0], label='h')
ax1.set_title('Cosine Similarity')
ax1.legend()

ax2.plot(wdiff_data[0], label='w')
ax2.plot(hdiff_data[0], label='h')
ax2.set_title('Weighted Difference')
ax2.legend()

ax3.plot(nds_data[0], label='h*')
ax3.plot(nd_data[0], label='h')
ax3.set_title('Norm')
ax3.legend()

plt.show()
exit()

# save things to disk
save_context = exp.buildSaveContext(idx)
save_context.ensureExists()

np.save(save_context.resolve('delta_sim_summary.npy'), np.array(hsim_data))
np.save(save_context.resolve('v_sim_summary.npy'), np.array(wsim_data))
np.save(save_context.resolve('delta_diff_summary.npy'), np.array(hdiff_data))
np.save(save_context.resolve('v_diff_summary.npy'), np.array(wdiff_data))
