import numpy as np
from src.agents.BaseTD import BaseTD

class BaseTDC(BaseTD):
    def tde_h(self, r, gamma, v_tp1, v_t, p):
        raise NotImplementedError()

    def tde_w(self, r, gamma, v_tp1, v_t, p):
        raise NotImplementedError()

    def dh(self, tde_h, z, h_dot_x, p):
        raise NotImplementedError()

    def computeGradient(self, obs_t, a_t, obs_tp1, r, gamma, p):
        w, h = self.theta
        v_tp1 = w.dot(obs_tp1)
        v_t = w.dot(obs_t)

        tde_h = self.tde_h(r, gamma, v_tp1, v_t, p)
        tde_w = self.tde_w(r, gamma, v_tp1, v_t, p)

        h_dot_x = h.dot(obs_t) * obs_t
        h_dot_z = h.dot(self.z) * obs_tp1
        # h_dot_z = self.h.dot(obs_tp1) * self.z
        # h_dot_z = self.z.dot(obs_tp1) * self.h


        dh = self.dh(tde_h, self.z, h_dot_x, p)
        dw = tde_w * self.z - p * gamma * (1 - self.lambdaa) * h_dot_z

        return [dw, dh]

# TDE
def correctWholeTDE(r, gamma, v_tp1, v_t, p):
    return p * (r + gamma * v_tp1 - v_t)

def correctTDTarget(r, gamma, v_tp1, v_t, p):
    return p * (r + gamma * v_tp1) - v_t

# H update
def correctHUpdate(tde_h, z, h_dot_x, p):
    return tde_h * z - p * h_dot_x

def noCorrectHUpdate(tde_h, z, h_dot_x, p):
    return tde_h * z - h_dot_x

# first:  correct H update
# second: correct tde_h
# third:  correct tde_w

def getTDCMethod(code):
    h_upd = [ correctHUpdate, noCorrectHUpdate ][int(code[0])]
    tde_h = [ correctWholeTDE, correctTDTarget ][int(code[1])]
    tde_w = [ correctWholeTDE, correctTDTarget ][int(code[2])]

    class TDCMethod(BaseTDC):
        def tde_h(self, r, gamma, v_tp1, v_t, p):
            return tde_h(r, gamma, v_tp1, v_t, p)

        def tde_w(self, r, gamma, v_tp1, v_t, p):
            return tde_w(r, gamma, v_tp1, v_t, p)

        def dh(self, tde_h, z, h_dot_x, p):
            return h_upd(tde_h, z, h_dot_x, p)

    return TDCMethod
