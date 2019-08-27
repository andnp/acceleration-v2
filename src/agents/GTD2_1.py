import numpy as np
from src.agents.BaseTD import BaseTD

# this learns h without importance sampling corrections
# this means that np.dot(h, x) ~ b instead of np.dot(h, x) ~ pi
class GTD2_1(BaseTD):
    def computeGradient(self, obs_t, a_t, obs_tp1, r, gamma, p):
        v_tp1 = self.w.dot(obs_tp1)
        v_t = self.w.dot(obs_t)

        tde_h = r + gamma * v_tp1 - v_t

        h_dot_x = self.h.dot(obs_t) * obs_t
        h_dot_z = self.h.dot(self.z) * obs_tp1

        # no importance sampling correction on h at all
        dh = tde_h * self.z - h_dot_x
        dw = p * (h_dot_x - gamma * (1.0 - self.lambdaa) * h_dot_z)

        return [dw, dh]
