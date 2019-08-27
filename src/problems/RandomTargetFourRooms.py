import numpy as np
from PyExpUtils.utils.fp import once
from PyFixedReps.TileCoder import TileCoder
from src.problems.BaseProblem import BaseProblem, StepModel
from src.environments.FourRooms import FourRooms
from src.utils.rlglue import OffPolicyWrapper
from src.utils.SampleGenerator import SampleGenerator
import src.utils.policies as Policy

class RandomTargetFourRooms(BaseProblem):
    def _buildRepresentation(self):
        return TileCoder({
            'dims': 2,
            'tiles': 2,
            'tilings': 16,
            'random_offset': True,
        })

    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.exp = exp
        self.idx = idx

        # build state distribution
        self.db = getStateDistribution() # TODO
        # build true value function
        self.v_star = np.zeros() # TODO

        # build representation
        self.rep = self._buildRepresentation()
        # build environment
        self.env = FourRooms()
        # build agent
        self.agent = self.Agent(self.rep.features(), self.metaParameters)

        # build behavior policy
        self.behavior = Policy.fromActionArray([0.25, 0.25, 0.25, 0.25])
        # build target policy
        self.target = Policy.fromActionArray([]) # TODO

        # compute the observable value for each state once
        self.all_observables = np.array([
            self.rep.encode([i, j]) for i in range(11) for j in range(11)
        ])

    def getGamma(self):
        return 0.9

    def getSteps(self):
        return 5000

    def getEnvironment(self):
        return self.env

    def getRepresentation(self):
        return self.rep

    def getAgent(self):
        # wrap the agent so that its API is consistent with the RLGlue API
        # this is just a compatibility layer between disparate APIs
        return OffPolicyWrapper(self.agent, self.behavior, self.target, self.rep.encode)

    def evaluateStep(self, step_data):
        # absolute distance from v_star
        d = self.agent.value(self.all_observables) - self.v_star

        if np.any(d > 5) or np.any(np.isnan(d)) or np.any(np.isinf(d)):
            return np.NaN

        # weighted sum over squared distances
        s = np.sum(self.db * np.square(d))

        return np.sqrt(s)

    def sampleExperiences(self):
        clone = RandomTargetFourRooms(self.exp, self.idx)
        clone.rep = self.rep
        gen = SampleGenerator(clone)
        return gen.sample(samples = 100)

@once
def getStateDistribution():
    # attempt to load db
    # if not found, assume we will create it first
    # TODO: generate if not found
    try:
        return np.genfromtxt(f'baselines/FourRooms_db.csv')
    except:
        print('db not found, must create it before running experiments')
        return None
