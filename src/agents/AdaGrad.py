import numpy as np
from src.agents.BaseTD import BaseTD

class AdaGrad(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)

        self.S = np.zeros((2, features))
        self.p = params.get("p_norm")
        if self.p is None:
            self.p = 2

    def _stepsize(self):
        # promote to an np.array just once to reduce unnecessary implicit casts by numpy
        dtheta = np.array(self.dtheta)
        S = self.S + np.square(dtheta)
        ss = self.stepsize / (np.sqrt(S) + 1e-8)

        # if the denominator has a 0, then we end up with massive spikes in stepsize
        # if those spikes occur where the update is also 0, then we aren't actually making a huge update
        # so to meaningfully visualize the stepsize, we can get rid of those spikes by making the stepsize 0 in those cases
        # which is safe because the update was also 0
        ss = np.where((S == 0) & (dtheta == 0), np.zeros_like(ss), ss)

        return np.mean(ss, axis=1)

    def update(self, x, a, xp, r, gamma, p):
        dtheta = self.computeGradient(x, a, xp, r, gamma, p)

        self.S = self.S + np.power(dtheta, self.p)

        self.theta = self.theta + (self.stepsize / (np.power(self.S,1.0/self.p) + 1e-8)) * dtheta

        self.last_p = p
        self.last_gamma = gamma
        self.dtheta = dtheta
