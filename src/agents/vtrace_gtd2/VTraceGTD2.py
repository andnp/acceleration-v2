import numpy as np
from src.agents.BaseTD import BaseTD

class VTraceGTD2(BaseTD):
    def computeGradient(self, x, a, xp, r, gamma, p):
        w, h = self.theta
        vp = w.dot(xp)
        v = w.dot(x)

        delta = r + gamma * vp - v
        delta_hat = h.dot(x)

        k = np.min((1, p))

        dw = k * (delta_hat * x - gamma * delta_hat * xp)
        dh = (k * delta - delta_hat) * x

        return [dw, dh]
