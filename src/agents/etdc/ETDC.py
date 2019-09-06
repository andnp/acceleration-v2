import numpy as np
from src.agents.BaseTD import BaseTD

class ETDC(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)

        self.f = 0

    def computeGradient(self, x, a, xp, r, gamma, p):
        w, h = self.theta
        vp = w.dot(xp)
        v = w.dot(x)

        delta = r + gamma * vp - v
        delta_hat = h.dot(x)

        dw = p * self.f * (delta * x - gamma * delta_hat * xp)
        dh = (p * self.f * delta - delta_hat) * x

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