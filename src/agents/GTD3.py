import numpy as np
from src.agents.BaseTD import BaseTD

# this doesn't work because w is unbounded and unconstrained
# I would need to find a way to make w part of the update
class GTD3(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)
        self.alpha_g = params['alpha_g']

        self.zg = np.zeros(features)
        self.theta = np.zeros((3, features))
        self.stepsize = np.tile([self.alpha, self.alpha_h, self.alpha_g], (features, 1)).T

    def nextTrace(self, obs_t):
        self.zg = self.last_gamma * self.lambdaa * self.zg + obs_t
        return self.last_p * self.last_gamma * self.lambdaa * self.z + obs_t

    def reset(self):
        self.zg = np.zeros(self.features)
        self.z = np.zeros(self.features)

    def computeGradient(self, obs_t, a_t, obs_tp1, r, gamma, p):
        w, h, g = self.theta

        v_tp1 = g.dot(obs_tp1)
        v_t = g.dot(obs_t)

        tde_g = r + gamma * v_tp1 - v_t

        h_dot_x = h.dot(obs_t) * obs_t

        # no importance sampling correction on h at all
        dh = tde_g * self.zg - h_dot_x
        dg = h_dot_x - gamma *(1.0 - self.lambdaa) * h.dot(self.zg) * obs_tp1
        dw = p * (h_dot_x - gamma * (1.0 - self.lambdaa) * h.dot(self.z) * obs_tp1)

        return [dw, dh, dg]
