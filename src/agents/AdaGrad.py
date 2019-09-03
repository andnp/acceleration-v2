import numpy as np
from src.agents.BaseTD import BaseTD

class AdaGrad(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)

        self.S = np.zeros((2, features))

    def _stepsize(self, dtheta):
        # promote to an np.array just once to reduce unnecessary implicit casts by numpy
        dtheta = np.array(dtheta)
        S = self.S + np.square(dtheta)
        ss = self.stepsize / (np.sqrt(S) + 1e-8)

        ss = np.where(dtheta > 0, ss, np.zeros_like(ss))

        return np.mean(ss, axis=1)

    def update(self, x, a, xp, r, gamma, p):
        dtheta = self.computeGradient(x, a, xp, r, gamma, p)

        self.S = self.S + np.square(dtheta)

        self.theta = self.theta + (self.stepsize / (np.sqrt(self.S) + 1e-8)) * dtheta

        self.last_p = p
        self.last_gamma = gamma
