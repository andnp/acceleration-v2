import numpy as np
from src.agents.BaseTD import BaseTD

class AdaGradPNorm(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)

        self.v1 = np.zeros((1, features))
        self.v2 = np.zeros((1, features))
        self.p = params.get("p_norm")
        if self.p is None:
            self.p = 2

    def update(self, x, a, xp, r, gamma, p):
        dw, dh = self.computeGradient(x, a, xp, r, gamma, p)

        self.v1 = self.v1 + np.square(dw)
        self.v2 = self.v2 + np.power(dh, self.p)

        self.theta[0] = self.theta[0] + (self.stepsize[0]/(np.sqrt(self.v1)+1e-8))*dw
        self.theta[1] = self.theta[1] + (self.stepsize[1] / (np.power(self.v2,1.0/self.p) + 1e-8)) * dh

        self.last_p = p
        self.last_gamma = gamma
