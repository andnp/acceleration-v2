import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.problems.BaseProblem import BaseProblem, StepModel
from src.environments.Spiral import Spiral as SpiralEnv
from src.utils.rlglue import OffPolicyWrapper

import src.utils.policies as Policies

N = 3

class Spiral(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx
        perm = exp.getPermutation(idx)

        self.env = SpiralEnv()

        self.nsteps = perm["nsteps"]

        # build representation
        self.rep = OneHot(3)

        # build environment
        # build agent
        self.agent = self.Agent(SpiralValueFunction(), 1, self.metaParameters)
        self.agent.theta[0] = np.array([0])

        self.target = Policies.fromActionArray([1.0])
        self.behavior = Policies.fromActionArray([1.0])

        # compute the observable value for each state once
        self.all_observables = np.array([
            self.rep.encode(i) for i in range(N)
        ])

        self.db=np.array([1/3,1/3,1/3])
        self.v_star = np.zeros(3)

        # build transition probability matrix (under target)
        self.P = np.zeros((N, N))
        for i in range(N):
            self.P[i,i] = 0.5
            self.P[i,i-1] = 0.5

        self.R = np.zeros(N)

        self.setupIdealH()

    def getTarget(self, n):
        return NotImplementedError()

    def getGamma(self):
        return 0.9

    def getSteps(self):
        return self.nsteps

    def getEnvironment(self):
        return self.env

    def getRepresentation(self):
        return self.rep

    def getAgent(self):
        return OffPolicyWrapper(self.agent, self.behavior, self.target, self.rep.encode)

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

class OneHot(BaseRepresentation):
    def __init__(self, N):
        self.map = np.zeros((N,N))

        idx = 0
        for i in range(N):
            self.map[i,i] = 1

    def encode(self, s):
        return self.map[s]

    def features(self):
        return self.map.shape[1]

class SpiralValueFunction:
    def __init__(self, lambda_hat=0.866, epsilon=0.05):
        self.eps = epsilon
        self.lmda = lambda_hat
        self.ab = np.array([[100,-70,-30],
                           [23.094,-98.15,75.056]])

    def eval(self, x, w):
        a,b = np.dot(self.ab, x)
        return np.exp(self.eps*w)*(a*np.cos(self.lmda*w)-b*np.sin(self.lmda*w))

    def Rop(self, x, w, h):
        a, b = np.dot(self.ab, x)
        eps,lmda = self.eps, self.lmda
        c, s = np.cos(w*lmda), np.sin(w*lmda)
        return h*np.exp(eps*w)*((a*eps*eps-2*eps*lmda*b-a*lmda*lmda)*c - (b*eps*eps+2*a*eps*lmda-b*lmda*lmda)*s)

