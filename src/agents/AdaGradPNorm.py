import numpy as np
from src.agents.BaseTD import BaseTD

class AdaGradPNorm(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)

        self.v1 = np.zeros((1, features))
        self.v2 = np.zeros((1, features))
        self.p = params.get("p_norm")

    def _stepsize(self):
        # promote to an np.array just once to reduce unnecessary implicit casts by numpy
        dtheta = np.array(self.dtheta)
        v1 = self.v1 + np.square(dtheta[0])
        v2 = self.v2 + np.power(dtheta[1], self.p)
        S = np.array([np.sqrt(v1),np.power(v2, 1.0/self.p)])
        ss = self.stepsize / (S + 1e-8)

        # if the denominator has a 0, then we end up with massive spikes in stepsize
        # if those spikes occur where the update is also 0, then we aren't actually making a huge update
        # so to meaningfully visualize the stepsize, we can get rid of those spikes by making the stepsize 0 in those cases
        # which is safe because the update was also 0
        ss = np.where((S == 0) & (dtheta == 0), np.zeros_like(ss), ss)

        return np.mean(ss, axis=1)

    def update(self, x, a, xp, r, gamma, p):
        dtheta = self.computeGradient(x, a, xp, r, gamma, p)
        dw,dh = dtheta

        self.v1 = self.v1 + np.square(dw)
        self.v2 = self.v2 + np.power(dh, self.p)

        self.theta[0] = self.theta[0] + (self.stepsize[0]/(np.sqrt(self.v1)+1e-8))*dw
        self.theta[1] = self.theta[1] + (self.stepsize[1] / (np.power(self.v2,1.0/self.p) + 1e-8)) * dh

        self.last_p = p
        self.last_gamma = gamma
        self.dtheta = dtheta
