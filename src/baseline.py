import numpy as np
from multiprocessing import Pool
import sys
import os
sys.path.append(os.getcwd())

from src.problems.StandardCollision import StandardCollision
from src.problems.BairdCounterexample import BairdCounterexample
from src.problems.BigRhoCollision import BigRhoCollision
from src.utils.model import loadExperiment

samples = int(1e7)

problems = [
    # (StandardCollision, 'StandardCollision'),
    (BigRhoCollision, 'BigRhoCollision'),
    # (BairdCounterexample, 'BairdCounterexample'),
]

def writePolicyBaseline(pair):
    Problem, name = pair
    problem = Problem(loadExperiment('./experiments/test.json'), 0)
    behaviorPolicy = problem.behavior
    targetPolicy = problem.target

    env = problem.getEnvironment()
    rep = problem.getRepresentation()
    dist = np.zeros(env.states)
    experiences = []
    s = env.start()
    for i in range(samples):
        if i % 100000 == 0:
            print(i / samples)
        a = behaviorPolicy.selectAction(s)
        r, sp, d = env.step(a)
        rho = targetPolicy.ratio(behaviorPolicy, s, a)
        gamma = 0 if d else problem.getGamma()

        ex = rep.encode(s), a, rep.encode(sp), r, gamma, rho
        experiences.append(ex)

        s = sp

        dist[sp] += 1

    dist /= samples
    print(dist)
    np.savetxt(f'baselines/{name}_db.csv', dist)
    # sampled_exp = np.random.choice(range(samples), size=10000)
    # experiences = np.array(experiences)
    # np.save(f'baselines/experiences_{name}.npy', experiences[sampled_exp])

if __name__ == '__main__':
    pool = Pool()
    pool.map(writePolicyBaseline, problems)
