import numpy as np
from src.agents.BaseTD import BaseTD

class HTD(BaseTD):
    def computeGradient(self, x, a, xp, r, gamma, p):
        w, h = self.theta
        vp = w.dot(xp)
        v = w.dot(x)

        delta = r + gamma * vp - v
        delta_hat = h.dot(x)

        dh = (p * delta * x - delta_hat * (x - gamma * xp))
        dw = p * delta * x + (x - gamma * xp) * (p - 1) * delta_hat

        return [dw, dh]
