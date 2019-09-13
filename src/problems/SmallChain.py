import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.problems.BaseProblem import BaseProblem, StepModel
from src.environments.Chain import Chain as ChainEnv
from src.utils.rlglue import OffPolicyWrapper
from src.utils.policies import Policy

class BaseChain(BaseProblem):
    def _getTarget(self):
        return Policy(lambda s: [])

    def _getRepresentation(self, n):
        return BaseRepresentation()

    def _getSize(self):
        return 5

    def _getdb(self):
        return np.array([0.111111, 0.222222, 0.333333, 0.222222, 0.111111])

    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx

        N = self._getSize()

        self.env = ChainEnv(N)

        # build representation
        self.rep = self._getRepresentation(N)

        # build environment
        # build agent
        self.agent = self.Agent(self.rep.features(), self.metaParameters)

        # build target policy
        self.target = self._getTarget()

        self.behavior = Policy(lambda s: [0.5, 0.5])

        # compute the observable value for each state once
        self.all_observables = np.array([
            self.rep.encode(i) for i in range(N)
        ])

        # (1/n+1) sum_{k=0}^n P^k gives a matrix with db in each row, where P is the markov chain
        # induced by the behaviour policy
        self.db = self._getdb()

        self.v_star = self.compute_v(N, self.target)

        # build transition probability matrix (under target)
        self.P = np.zeros((N, N))
        pl, pr = self.target.probs(0)
        self.P[0, 1] = pr
        self.P[0, N // 2] = pl
        self.P[N-1, N-2] = pl
        self.P[N-1, N // 2] = pr
        for i in range(1, N-1):
            self.P[i, i - 1] = pl
            self.P[i, i + 1] = pr

        self.R = np.zeros(N)
        self.R[0] = pl * -1
        self.R[N-1] = pr * 1

        self.setupIdealH()

    def getTarget(self, n):
        return NotImplementedError()

    def getGamma(self):
        return 1.0

    def getSteps(self):
        return 1000

    def getEnvironment(self):
        return self.env

    def getRepresentation(self):
        return self.rep

    def getAgent(self):
        return OffPolicyWrapper(self.agent, self.behavior, self.target, self.rep.encode)

    def compute_v(self, nstates, targetPolicy):
        gamma = self.getGamma()
        theta = 1e-8

        V = np.zeros(nstates)

        delta = np.infty
        i = 0
        while delta > theta:
            i += 1
            delta = 0.0
            for s in range(nstates):
                p_left, p_right = targetPolicy.probs(s)

                v = V[s]

                right = s + 1
                if right >= nstates:
                    right_reward = 1
                    right_value = 0
                else:
                    right_reward = 0.0
                    right_value = V[right]

                left = s - 1
                if left<0:
                    left_reward = -1.0
                    left_value = 0.0
                else:
                    left_reward = 0.0
                    left_value = V[left]

                V[s] = p_right * (right_reward + gamma * right_value) +\
                       p_left * (left_reward + gamma * left_value)

                delta = max(delta, np.abs(v - V[s]))

        return V

    def evaluateStep(self, step_data):
        # distance from v_pi
        d = self.agent.value(self.all_observables) - self.v_star
        # weighted sum over squared distances
        s = np.sum(self.db * np.square(d))

        rmsve = np.sqrt(s)

        w = self.agent.theta[0]
        A = self.A
        b = self.b
        C = self.C

        v = np.dot(-A, w) + b
        mspbe = v.T.dot(np.linalg.pinv(C)).dot(v)
        rmspbe = np.sqrt(mspbe)

        return rmsve, rmspbe

# ----------------
# -- On-policy --
# ----------------

class Chain5050(BaseChain):
    def _getTarget(self):
        return Policy(lambda s: [.5, .5])

# --------------------
# -- Representation --
# --------------------

class ChainInverted(BaseChain):
    def _getRepresentation(self, n):
        return Inverted(n)

class ChainTabular(BaseChain):
    def _getRepresentation(self, n):
        return Tabular(n)

class ChainDependent(BaseChain):
    def _getRepresentation(self, n):
        return Dependent(n)

# ---------------
# -- Resultant --
# ---------------

class SmallChainDependent5050(Chain5050, ChainDependent):
    pass

class SmallChainInverted5050(Chain5050, ChainInverted):
    pass
    
class SmallChainTabular5050(Chain5050, ChainTabular):
    pass



# --------------------
# -- Representation --
# --------------------

class Inverted(BaseRepresentation):
    def __init__(self, N):
        self.map = np.ones((N,N)) - np.eye(N)

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[0]

class Tabular(BaseRepresentation):
    def __init__(self, N):
        self.map = np.eye(N)

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]

class Dependent(BaseRepresentation):
    def __init__(self, N):
        nfeats = int(np.floor(N/2) + 1)
        self.map = np.zeros((N,nfeats))

        idx = 0
        for i in range(nfeats):
            self.map[idx,0:i+1] = 1
            idx+=1

        for i in range(nfeats-1,0,-1):
            self.map[idx,-i:] = 1
            idx+=1

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]
