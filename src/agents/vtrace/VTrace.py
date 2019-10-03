import numpy as np
from src.agents.BaseTD import BaseTD

class VTrace(BaseTD):
    def computeGradient(self, obs_t, a_t, obs_tp1, r, gamma, p):
        w = self.theta[0]
        v_tp1 = np.dot(obs_tp1, w)
        v_t = np.dot(obs_t, w)

        delta = np.min(p, 1) * (r + gamma * v_tp1 - v_t)

        dw = delta * obs_t
        dh = np.zeros(self.features)
        return [ dw, dh ]
