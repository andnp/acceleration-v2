import numpy as np
from src.agents.BaseTD import BaseTD

class Schedule(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)

        self.n0 = params['n0']
        self.steps = 0

    def _stepsize(self):
        schedule = (self.n0 + 1) / (self.n0 + self.steps)
        alpha = self.alpha * schedule
        alpha_h = self.alpha_h * schedule

        return (alpha, alpha_h)

    def update(self, x, a, xp, r, gamma, p):
        dtheta = self.computeGradient(x, a, xp, r, gamma, p)

        schedule = (self.n0 + 1) / (self.n0 + self.steps)
        alpha = self.alpha * schedule
        alpha_h = self.alpha_h * schedule

        stepsize = np.tile([alpha, alpha_h], (self.features, 1)).T

        self.theta = self.theta + stepsize * dtheta

        self.last_p = p
        self.last_gamma = gamma
        self.dtheta = dtheta
        self.steps += 1
