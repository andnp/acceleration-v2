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

        # collect effective stepsize
        ss_w, ss_h = problem.agent._stepsize()
        if is_using_ideal_h:
            ss_h = 0

        collector.collect('stepsize', [ss_w, ss_h])

        # collect the norm of h
        if is_using_ideal_h:
            h = agent_wrapper.agent.getIdealH()
        else:
            h = agent_wrapper.agent.theta[1]

        h_norm = np.linalg.norm(h)
        collector.collect('hnorm', h_norm)

        # collect h update sizes
        if is_using_ideal_h:
            dh = np.zeros_like(h)
        else:
            dh = agent_wrapper.agent.dtheta[1]

        update_size = np.linalg.norm(ss_h * dh)
        collector.collect('h_update', update_size)

        # ||np.dot(h, X)|| should go to zero
        X = problem.all_observables
        norm_delta_hat = weightedNorm(np.dot(X, h), problem.db)
        collector.collect('norm_delta_hat', norm_delta_hat)

        # if we've diverged, just go ahead and give up
        # saves some computation and these runs are useless to me anyways
        if np.isnan(rmsve) or np.isinf(rmsve):
            collector.fillRest(np.nan, problem.getSteps())
            broke = True
            break

    collector.reset()
    if broke:
        break


# get stats over runs for each collected variable
error_data = collector.getStats('errors')
rmspbe_data = collector.getStats('rmspbe')
ss_data = collector.getStats('stepsize')
hnorm_data = collector.getStats('hnorm')
hupd_data = collector.getStats('h_update')
ndh_data = collector.getStats('norm_delta_hat')

# local plotting (for testing)
# fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

# ax1.plot(error_data[0])
# ax1.set_title('RMSVE')
# ax2.plot(rmspbe_data[0])
# ax2.set_title('RMSPBE')

# for m, label in zip(ss_data[0].T, ['w', 'h']):
#     ax3.plot(m, label=label)
# ax3.legend()
# ax3.set_title('stepsize')

# ax4.plot(hnorm_data[0])
# ax4.set_title('hnorm')
# ax5.plot(hupd_data[0])
# ax5.set_title('h update')
# ax6.plot(ndh_data[0])
# ax6.set_title('norm of delta_hat')

# plt.show()
# exit()

# save things to disk
save_context = exp.buildSaveContext(idx)
save_context.ensureExists()

np.save(save_context.resolve('errors_summary.npy'), np.array(error_data))
np.save(save_context.resolve('rmspbe_summary.npy'), np.array(rmspbe_data))
np.save(save_context.resolve('stepsize_summary.npy'), np.array(ss_data))
np.save(save_context.resolve('hnorm_summary.npy'), np.array(hnorm_data))
np.save(save_context.resolve('hupd_summary.npy'), np.array(hupd_data))
np.save(save_context.resolve('ndh_summary.npy'), np.array(ndh_data))
