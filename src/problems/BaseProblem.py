import numpy as np
from src.agents.agents import getAgent

def weightedNorm(X, W):
    return np.sqrt(X.T.dot(W).dot(X))

class BaseProblem:
    def __init__(self, exp, idx):
        self.Agent = getAgent(exp.agent)

        # what parameter permutation should we use
        perm = exp.getPermutation(idx)

        # add gamma to the agent parameters
        # gamma is a problem parameter, but the agent needs access to it
        self.metaParameters = perm['metaParameters']
        self.metaParameters['gamma'] = self.getGamma()

    def getEnvironment(self):
        raise NotImplementedError()

    def getRepresentation(self):
        raise NotImplementedError()

    def getAgent(self):
        raise NotImplementedError()

    def getGamma(self):
        raise NotImplementedError()

    def getSteps(self):
        raise NotImplementedError()

    def sampleExperiences(self):
        raise NotImplementedError()

    def evaluateStep(self, step_data):
        X = getattr(self, 'X')
        # absolute distance from v_star
        d = self.agent.value(X) - self.v_star

        # weighted sum over squared distances
        rmsve = weightedNorm(d, np.diag(self.db))

        w = self.agent.theta[0]
        A = self.A
        b = self.b
        Cinv = self.Cinv

        v = np.dot(-A, w) + b
        mspbe = v.T.dot(Cinv).dot(v)
        rmspbe = np.sqrt(mspbe)

        return rmsve, rmspbe

    def evaluateEpisode(self, episode):
        pass

    def setupIdealH(self):
        # TODO(andy): make this less attrocious
        X = getattr(self, 'X')
        dB = np.diag(getattr(self, 'db'))
        gamma = self.getGamma()
        P = getattr(self, 'P')
        R = getattr(self, 'R')

        A = X.T.dot(dB).dot(np.eye(X.shape[0]) - gamma * P).dot(X)
        b = X.T.dot(dB).dot(R)
        C = X.T.dot(dB).dot(X)

        self.A = A
        self.b = b
        self.C = C
        self.Cinv = np.linalg.pinv(C)

        agent_wrapper = self.getAgent()
        # well this sucks. first agent is the off-policy-wrapper
        # second agent is the actual TD agent
        agent_wrapper.agent.ideal_h_params = (A, b, C)
