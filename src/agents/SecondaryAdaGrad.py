import numpy as np
from src.agents.BaseTD import BaseTD

class SecondaryAdaGrad(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)

        self.S = np.zeros((1,features))

    def update(self, x, a, xp, r, gamma, p):
        dw,dh = self.computeGradient(x, a, xp, r, gamma, p)

        self.S = self.S + np.square(dh)

        self.theta[0] = self.theta[0] + self.alpha * dw
        self.theta[1] = self.theta[1] + (self.alpha_h / (np.sqrt(self.S)+1e-8))*dh

        self.last_p = p
        self.last_gamma = gamma
