import numpy as np
from src.problems.BairdCounterexample import BairdCounterexample
from src.environments.Baird import Baird
from src.utils.rlglue import OffPolicyWrapper
from src.utils.SampleGenerator import SampleGenerator


import src.utils.policies as Policy

class BairdRMSPBE(BairdCounterexample):
    def __init__(self, exp, idx):
        super().__init__(exp, idx) # This calls bairdcounterexample init method.
        self.setupIdealH()

    def evaluateStep(self, step_data):
        w = self.agent.theta[0]
        A = self.A
        b = self.b
        C = self.C
        
        v = np.dot(-A, w) + b
        MSPBE = v.T.dot(np.linalg.pinv(C)).dot(v)

        return np.sqrt(MSPBE)