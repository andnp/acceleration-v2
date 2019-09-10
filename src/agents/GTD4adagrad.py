import numpy as np
from src.agents.BaseTD import BaseTD
from src.agents.optimizers.AdaGrad import AdaGrad


class GTD4(AdaGrad):
    def computeGradient(self, x, a, xp, r, gamma, p):
        w, h = self.theta
        vp = w.dot(xp)
        v = w.dot(x)

        delta = r + gamma * vp - v
        delta_hat = h.dot(x)

        dh = (p * delta - delta_hat) * x
        dw = (delta * x - gamma * delta_hat * xp)

        return [dw, dh]
