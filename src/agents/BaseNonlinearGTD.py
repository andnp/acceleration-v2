import numpy as np
from .BaseTD import BaseTD

class FunctionApproximator:
    def __init__(self):
        pass

    def eval(self, x, w):
        return NotImplementedError()

    def Rop(self, x, w, h):
        # Compute the product of Hessian(V)h using an R-operator
        return NotImplementedError()

class Spiral(FunctionApproximator):
    def __init__(self, lambda_hat=0.866, epsilon=0.05):
        self.eps = epsilon
        self.lmda = lambda_hat
        self.ab = np.array([[100,-70,-30],
                           [23.094,-98.15,75.056]])

    def eval(self, x, w):
        a,b = np.dot(self.ab, x)
        return np.exp(self.eps*w)*(a*np.cos(self.lmda*w)-b*np.sin(self.lmda*w))

    def Rop(self, x, w, h):
        a, b = np.dot(self.ab, x)
        eps,lmda = self.epsilon, self.lmda
        c, s = np.cos(w*lmda), np.sin(w*lmda)
        return h*np.exp(eps*w)*((a*np.pow(eps,2)-2*eps*lmda*b-a*np.pow(lmda,2))*c - (b*np.pow(eps,2)+2*a*eps*lmda-b*np.pow(lmda,2))*s)


class NonlinearGTD2(BaseTD):
    def __init__(self, value_fn, features, params):
        super().__init__(features, params)

        self.V = v

    def computeGradient(self, x_t, a_t, x_tp1, r, gamma, p):
        w,h = self.theta
        v_tp1 = self.V.eval(x_tp1,w)
        v_t = self.V.eval(x_t,w)

        delta = p * (r + gamma * v_tp1 - v_t)
        delta_hat = x_t.dot(h)
        delta_diff = delta-delta_hat

        dw = p * (delta_hat * x - gamma * delta_hat * xp)
        dw -= delta_diff*self.V.Rop(x_t,w,h)
        dh = (p * delta - delta_hat) * x
        return [ dw-delta_diff*self.V(x_t,w,h), dh ]

    def update(self, obs_t, a_t, obs_tp1, r, gamma, p):
        dtheta = self.computeGradient(obs_t, a_t, obs_tp1, r, gamma, p)

        # print(self.stepsize / (np.sqrt(self.S) + 1e-8))
        self.theta = self.theta + self.stepsize * dtheta

        self.last_p = p
        self.last_gamma = gamma
        self.dtheta = dtheta

    def reset(self):
        pass

    def value(self, obs):
        w = self.theta[0]
        return np.dot(obs, w)

class NonlinearTDC(BaseTD):
    def __init__(self, value_fn, features, params):
        super().__init__(features, params)

        self.V = v

    def computeGradient(self, x_t, a_t, x_tp1, r, gamma, p):
        w,h = self.theta
        v_tp1 = self.V.eval(x_tp1,w)
        v_t = self.V.eval(x_t,w)

        delta = p * (r + gamma * v_tp1 - v_t)
        delta_hat = x_t.dot(h)
        delta_diff = delta-delta_hat

        dw = p * (delta * x - gamma * delta_hat * xp)
        dw -= delta_diff*self.V.Rop(x_t,w,h)
        dh = (p * delta - delta_hat) * x
        return [ dw, dh ]

    def update(self, obs_t, a_t, obs_tp1, r, gamma, p):
        dtheta = self.computeGradient(obs_t, a_t, obs_tp1, r, gamma, p)

        # print(self.stepsize / (np.sqrt(self.S) + 1e-8))
        self.theta = self.theta + self.stepsize * dtheta

        self.last_p = p
        self.last_gamma = gamma
        self.dtheta = dtheta

    def reset(self):
        pass

    def value(self, obs):
        w = self.theta[0]
        return np.dot(obs, w)
