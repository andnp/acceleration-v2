import numpy as np
from src.agents.BaseTD import BaseTD
from src.utils.buffers import CircularBuffer

class ReghGTD2(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)

        self.reg_h = params['reg_h']

    def computeGradient(self, x, a, xp, r, gamma, p):
        w, h = self.theta
        vp = w.dot(xp)
        v = w.dot(x)

        delta = r + gamma * vp - v
        delta_hat = h.dot(x)

        dw = p * (delta_hat * x - gamma * delta_hat * xp)
        dh = (p * delta - delta_hat) * x - self.reg_h * h

        return [dw, dh]
