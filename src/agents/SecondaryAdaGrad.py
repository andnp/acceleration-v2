import numpy as np
from src.agents.BaseTD import BaseTD

class SecondaryAdaGrad(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)

        self.S = np.zeros((1,features))
        self._ones = np.ones_like(self.S)

    def _stepsize(self):
        # promote to an np.array just once to reduce unnecessary implicit casts by numpy
        dtheta = np.array(self.dtheta)
        S = self.S + np.square(dtheta[1])
        ss = self.stepsize / np.array([self._ones, np.sqrt(S)+1e-8])

        # if the denominator has a 0, then we end up with massive spikes in stepsize
        # if those spikes occur where the update is also 0, then we aren't actually making a huge update
        # so to meaningfully visualize the stepsize, we can get rid of those spikes by making the stepsize 0 in those cases
        # which is safe because the update was also 0
        ss = np.where((S == 0) & (dtheta == 0), np.zeros_like(ss), ss)

        return np.mean(ss, axis=1)

    def update(self, x, a, xp, r, gamma, p):
        dtheta = self.computeGradient(x, a, xp, r, gamma, p)
        dw,dh = dtheta

        self.S = self.S + np.square(dh)

        self.theta[0] = self.theta[0] + self.alpha * dw
        self.theta[1] = self.theta[1] + (self.alpha_h / (np.sqrt(self.S)+1e-8))*dh

        self.last_p = p
        self.last_gamma = gamma
        self.dtheta = dtheta
