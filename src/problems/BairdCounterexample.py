import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.problems.BaseProblem import BaseProblem, StepModel
from src.environments.Baird import Baird
from src.utils.rlglue import OffPolicyWrapper

import src.utils.policies as Policy

class BairdRep(BaseRepresentation):
    def __init__(self):
        self.map = np.array([
            [1, 2, 0, 0, 0, 0, 0, 0],
            [1, 0, 2, 0, 0, 0, 0, 0],
            [1, 0, 0, 2, 0, 0, 0, 0],
            [1, 0, 0, 0, 2, 0, 0, 0],
            [1, 0, 0, 0, 0, 2, 0, 0],
            [1, 0, 0, 0, 0, 0, 2, 0],
            [2, 0, 0, 0, 0, 0, 0, 1],
        ])

    def encode(self, s):
        return self.map[s]

    def features(self):
        return 8

class BairdCounterexample(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        # build state distribution
        self.db = np.ones(7) * (1/7)
        # build true value function
        self.v_star = np.zeros(7)

        # build representation
        self.rep = BairdRep()
        # build environment
        self.env = Baird()
        # build agent
        self.agent = self.Agent(self.rep.features(), self.metaParameters)

        # build behavior policy
        self.behavior = Policy.fromActionArray([6/7, 1/7])
        # build target policy
        self.target = Policy.fromActionArray([0.0, 1.0])

        # initialize agent with starting weight parameters
        self.agent.w = np.array([1, 1, 1, 1, 1, 1, 1, 10])

        # compute the observable value for each state once
        self.all_observables = np.array([
            self.rep.encode(i) for i in range(7)
        ])

    def getGamma(self):
        return 0.99

    def getSteps(self):
        return 1000

    def getEnvironment(self):
        return self.env

    def getRepresentation(self):
        return self.rep

    def getAgent(self):
        return OffPolicyWrapper(self.agent, self.behavior, self.target, self.rep.encode)

    def evaluateStep(self, step_data):
        # absolute distance from v_star
        d = self.agent.value(self.all_observables) - self.v_star

        # weighted sum over squared distances
        s = np.sum(self.db * np.square(d))

        return np.sqrt(s)

    def sampleExperiences(self):
        return np.load('baselines/experiences_BairdCounterexample.npy', allow_pickle=True)
