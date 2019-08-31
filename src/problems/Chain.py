import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.problems.BaseProblem import BaseProblem, StepModel
from src.environments.Chain import Chain as ChainEnv
from src.utils.rlglue import OffPolicyWrapper

import src.utils.policies as Policies

class Chain(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx
        perm = exp.getPermutation(idx)

        N = perm["nstates"]
        self.env = ChainEnv(N)

        self.nsteps = perm["nsteps"]

        # build representation
        self.rep = globals()[perm["representation"]](N)

        # build environment
        # build agent
        self.agent = self.Agent(self.rep.features(), self.metaParameters)

        # build target policy
        self.target = Policies.fromStateArray(
            [perm["target_policy"]]*N
        )

        # on-policy version of this domain
        self.behavior = Policies.fromStateArray(
            [[0.5,0.5]]*N
        )

        # compute the observable value for each state once
        self.all_observables = np.array([
            self.rep.encode(i) for i in range(N)
        ])

        # build transition probability matrix for computing ideal H
        self.P = np.zeros((N+1, N+1))
        for i in range(1,N):
            self.P[i, i-1] = .5
            self.P[i, i+1] = .5
        self.P[0,0] = 1.0
        self.P[N,N] = 1.0

        # build Reward structure for computing ideal H
        self.R = np.zeros(N+2)
        self.R[0] = -1.0
        self.R[N] = 1.0

        self.v_pi = self.compute_v(N, self.target)

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
        state_prob = 0.5
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
                if right > nstates-1:
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

                V[s] = state_prob * (
                    p_right * (right_reward + gamma*right_value) +
                    p_left * (left_reward + gamma*left_value)
                )

                delta = max(delta, np.abs(v-V[s]))
        return V

    def evaluateStep(self, step_data):
        # distance from v_pi
        d = self.agent.value(self.all_observables) - self.v_pi
        s = np.mean(np.square(d))
        return np.sqrt(s)

class OneHot(BaseRepresentation):
    def __init__(self, N):
        self.map = np.zeros((N,N))
        for i in range(N):
            self.map[i,i] = 1.0

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[0]

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
