import numpy as np
from src.agents.BaseTD import BaseTD

class GTD2(BaseTD):
    def computeGradient(self, x, a, xp, r, gamma, p):
        w, h = self.theta
        vp = w.dot(xp)
        v = w.dot(x)

        delta = p * (r + gamma * vp - v)
        delta_hat = h.dot(x)

        dw = delta_hat * x - p * gamma * delta_hat * xp
        dh = (delta - delta_hat) * x

        return [dw, dh]
