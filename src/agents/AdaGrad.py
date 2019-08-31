import numpy as np
from src.agents.BaseTD import BaseTD

class AdaGrad(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)

        self.S = np.zeros((2, features))

    def update(self, x, a, xp, r, gamma, p):
        dtheta = self.computeGradient(x, a, xp, r, gamma, p)

        self.S = self.S + np.square(dtheta)

        self.theta = self.theta + (self.stepsize / (np.sqrt(self.S) + 1e-8)) * dtheta

        self.last_p = p
        self.last_gamma = gamma
