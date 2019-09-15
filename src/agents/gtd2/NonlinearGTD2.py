import numpy as np
from src.agents.gtd2.GTD2 import GTD2

class NonlinearGTD2(GTD2):
    def __init__(self, value_fn, features, params):
        super().__init__(features, params)

        self.V = value_fn

    def computeGradient(self, x_t, a_t, x_tp1, r, gamma, p):
        w,h = self.theta
        v_tp1 = self.V.eval(x_tp1,w)
        v_t = self.V.eval(x_t,w)

        delta = p * (r + gamma * v_tp1 - v_t)
        delta_hat = x_t.dot(h)
        delta_diff = delta-delta_hat

        dw = p * (delta_hat * x_t - gamma * delta_hat * x_tp1)
        dw -= delta_diff*self.V.Rop(x_t,w,h)
        dh = (p * delta - delta_hat) * x_t
        return [ dw, dh ]
