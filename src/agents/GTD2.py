import numpy as np
from src.agents.BaseTD import BaseTD

class BaseGTD2(BaseTD):
    def tde_h(self, r, gamma, v_tp1, v_t, p):
        raise NotImplementedError()

    def dh(self, tde_h, z, h_dot_x, p):
        raise NotImplementedError()

    def dw(self, h_dot_x, gamma, h_dot_z, p):
        raise NotImplementedError()

    def computeGradient(self, obs_t, a_t, obs_tp1, r, gamma, p):
        w, h = self.theta
        v_tp1 = w.dot(obs_tp1)
        v_t = w.dot(obs_t)

        tde_h = self.tde_h(r, gamma, v_tp1, v_t, p)

        h_dot_x = h.dot(obs_t) * obs_t
        h_dot_z = h.dot(self.z) * obs_tp1

        dw = self.dw(h_dot_x, gamma, h_dot_z, p)
        dh = self.dh(tde_h, self.z, h_dot_x, p)

        return [dw, dh]

# H update
def correctHUpdate(tde_h, z, h_dot_x, p):
    return tde_h * z - p * h_dot_x

def noCorrectHUpdate(tde_h, z, h_dot_x, p):
    return tde_h * z - h_dot_x

# TDE
def correctWholeTDE(r, gamma, v_tp1, v_t, p):
    return p * (r + gamma * v_tp1 - v_t)

def correctTDTarget(r, gamma, v_tp1, v_t, p):
    return p * (r + gamma * v_tp1) - v_t

# W update
def correctWUpdate(h_dot_x, gamma, h_dot_z, lambdaa, p):
    return p * (h_dot_x - gamma * (1.0 - lambdaa) * h_dot_z)

def noCorrectWUpdate(h_dot_x, gamma, h_dot_z, lambdaa, p):
    return h_dot_x - p * gamma * (1.0 - lambdaa) * h_dot_z

# first:  correct H update
# second: correct tde_h
# third:  correct W update

def getGTD2Method(code):
    h_upd = [ correctHUpdate, noCorrectHUpdate ][int(code[0])]
    tde_h = [ correctWholeTDE, correctTDTarget ][int(code[1])]
    w_upd = [ correctWUpdate, noCorrectWUpdate ][int(code[2])]

    class GTD2Method(BaseGTD2):
        def dh(self, tde_h, z, h_dot_x, p):
            return h_upd(tde_h, z, h_dot_x, p)

        def tde_h(self, r, gamma, v_tp1, v_t, p):
            return tde_h(r, gamma, v_tp1, v_t, p)

        def dw(self, h_dot_x, gamma, h_dot_z, p):
            return w_upd(h_dot_x, gamma, h_dot_z, self.lambdaa, p)

    return GTD2Method
