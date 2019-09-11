import numpy as np
from src.agents.BaseTD import BaseTD
from src.agents.optimizers.AdaGrad import AdaGrad


class GTD6(AdaGrad):
    def computeGradient(self, x, a, xp, r, gamma, p):
        w, h = self.theta
        vp = w.dot(xp)
        v = w.dot(x)

        delta = r + gamma * vp - v
        delta_hat = h.dot(x)

        dh = (delta - delta_hat) * x
        dw = p * (delta_hat * x - gamma * delta_hat * xp)

        return [dw, dh]
