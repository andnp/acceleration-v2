import numpy as np
from src.agents.gtd2.GTD2 import GTD2

class NonlinearGTD2(GTD2):
    def __init__(self, value_fn, features, params):
        super().__init__(features, params)

        self.V = value_fn

    def value(self, s):
        return self.V.eval(s, self.theta[0])

    def computeGradient(self, s_t, a_t, s_tp1, r, gamma, p):
        w,h = self.theta
        x_t, x_tp1 = self.V.grad(s_t, w), self.V.grad(s_tp1, w)
        v_t, v_tp1 = self.V.eval(s_t,w), self.V.eval(s_tp1,w)

        delta = p * (r + gamma * v_tp1 - v_t)
        delta_hat = x_t.dot(h)
        delta_diff = delta-delta_hat

        dw = p * (delta_hat * x_t - gamma * delta_hat * x_tp1)
        dw -= delta_diff*self.V.Rop(s_t,w,h)
        dh = (p * delta - delta_hat) * x_t
        return [ dw, dh ]
