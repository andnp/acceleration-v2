import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.problems.BaseProblem import BaseProblem, StepModel
from src.environments.Chain import Chain as ChainEnv
from src.utils.rlglue import OffPolicyWrapper

import src.utils.policies as Policies

class BaseChain(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx
        perm = exp.getPermutation(idx)

        N = 19
        self.env = ChainEnv(N)

        self.nsteps = perm["nsteps"]

        # build representation
        self.rep = globals()[perm["representation"]](N)

        # build environment
        # build agent
        self.agent = self.Agent(self.rep.features(), self.metaParameters)

        # build target policy
        self.target = self.getTarget(N)

        self.behavior = Policies.fromStateArray(
            [[0.5,0.5]]*N
        )

        # compute the observable value for each state once
        self.all_observables = np.array([
            self.rep.encode(i) for i in range(N)
        ])

        # (1/n+1) sum_{k=0}^n P^k gives a matrix with db in each row, where P is the markov chain
        # induced by the behaviour policy
        self.db=np.array([0.0099983, 0.019997, 0.0299957, 0.0399956, 0.0499955, 0.0599974,
                          0.0699993, 0.080004, 0.0900087, 0.100017, 0.0900087, 0.080004,
                          0.0699993, 0.0599974, 0.0499955, 0.0399956, 0.0299957, 0.019997,
                          0.0099983])

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
        return self.nsteps

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
        i=0
        while delta > theta:
            i+=1
            delta = 0.0
            for s in range(nstates):
                p_left, p_right = targetPolicy.probs(s)

                v = V[s]

                right=s+1
                if right >= nstates:
                    right_reward=1
                    right_value=0
                else:
                    right_reward = 0.0
                    right_value = V[right]

                left = s-1
                if left<0:
                    left_reward = -1.0
                    left_value = 0.0
                else:
                    left_reward = 0.0
                    left_value = V[left]

                V[s] = p_right * (right_reward + gamma*right_value) +\
                       p_left * (left_reward + gamma*left_value)

                delta = max(delta, np.abs(v-V[s]))
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

class Chain4060(BaseChain):
    def getTarget(self,N):
        return Policies.fromStateArray(
            [[0.4,0.6]]*N
        )

class Chain2575(BaseChain):
    def getTarget(self,N):
        return Policies.fromStateArray(
            [[0.25,0.75]]*N
        )

class OneHot(BaseRepresentation):
    def __init__(self, N):
        self.map = np.zeros((N,N))
        for i in range(N):
            self.map[i,i] = 1.0

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]

class OneHotRedundant(BaseRepresentation):
    def __init__(self, N):
        self.map = np.zeros((N,N+1))
        for i in range(N):
            self.map[i,i] = 1.0
            self.map[i,-1] = 1.0

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]

class Inverted(BaseRepresentation):
    def __init__(self, N):
        self.map = np.ones((N,N))
        for i in range(N):
            self.map[i,i] = 0.0

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[0]

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
