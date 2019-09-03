import numpy as np
from src.agents.BaseTD import BaseTD

class TDC(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)

        self.use_ideal_h = params.get('use_ideal_h', False)

    def computeGradient(self, x, a, xp, r, gamma, p):
        w, h = self.theta
        vp = w.dot(xp)
        v = w.dot(x)

        if self.use_ideal_h:
            h = self.getIdealH()

        delta = p * (r + gamma * vp - v)
        delta_hat = h.dot(x)

        self.dw = delta * x - gamma * delta_hat * xp
        dh = (delta - delta_hat) * x

        return [self.dw, dh]
