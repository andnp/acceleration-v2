import numpy as np
from src.agents.BaseTD import BaseTD

class ETD(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)
        
        self.f = 0

    def computeGradient(self, x, a, xp, r, gamma, p):
        w = self.theta[0]
        v_tp1 = np.dot(xp, w)
        v_t = np.dot(x, w)

        delta = p * (r + gamma * v_tp1 - v_t)

        dw = delta * x * self.f
        dh = np.zeros(self.features)

        return [dw, dh]


    def update(self, obs_t, a_t, obs_tp1, r, gamma, p):
        self.f = self.last_p * gamma * self.f + 1
        dtheta = self.computeGradient(obs_t, a_t, obs_tp1, r, gamma, p)

        # print(self.stepsize / (np.sqrt(self.S) + 1e-8))
        self.theta = self.theta + self.stepsize * dtheta

        self.last_p = p
        self.last_gamma = gamma
        self.dtheta = dtheta

    def reset(self):
        self.f = 0
        self.last_p = 0