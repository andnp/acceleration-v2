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
        self.rep = Tabular(3)

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
        d = [self.agent.value(self.all_observables[i])-self.v_star[i] for i in range(N)]
        # weighted sum over squared distances
        s = np.sum(self.db * np.square(d))

        rmsve = np.sqrt(s)

        w = self.agent.theta[0]
        A = self.A
        b = self.b
        C = self.C

        v = np.dot(-A, w) + b
        mspbe = v.dot(v) / C
        rmspbe = np.sqrt(mspbe)

        return rmsve, rmspbe

class Tabular(BaseRepresentation):
    def __init__(self, N):
        self.N = N

    def encode(self, s):
        return s

    def features(self):
        return 1

class SpiralValueFunction:
    def __init__(self, lambda_hat=0.866, epsilon=0.05):
        self.eps = epsilon
        self.lmda = lambda_hat
        self.a = np.array([100,-70,-30])
        self.b = np.array([23.094,-98.15,75.056])

    def eval(self, s, w):
        a,b = self.getCoeffs(s)
        return np.exp(self.eps*w)*(a*np.cos(self.lmda*w)-b*np.sin(self.lmda*w))

    def getCoeffs(self, s):
        return self.a[s], self.b[s]

    def grad(self, s, w):
        a,b = self.getCoeffs(s)
        eps,lmda = self.eps, self.lmda
        return np.array([np.exp(eps*w)*((eps*a-b*lmda)*np.cos(lmda*w)-(eps*b+a*lmda)*np.sin(lmda*w))])

    def Rop(self, s, w, h):
        a,b = self.getCoeffs(s)
        eps,lmda = self.eps, self.lmda
        cos, sin = np.cos(w*lmda), np.sin(w*lmda)
        return np.array([h*np.exp(eps*w)*((a*eps*eps-2*eps*lmda*b-a*lmda*lmda)*cos - (b*eps*eps+2*a*eps*lmda-b*lmda*lmda)*sin)])

