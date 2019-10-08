import numpy as np
from src.agents.BaseTD import BaseTD

class AMSGrad(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)

        self.momentum = params['momentum']
        self.curve = params['curve']

        self.M = np.zeros((2, features))
        self.S = np.zeros((2, features))
        self.v = np.ones((2, features)) * 1e-8

    def _stepsize(self):
        # promote to an np.array just once to reduce unnecessary implicit casts by numpy
        dtheta = np.array(self.dtheta)
        M = self.momentum * self.M + (1 - self.momentum) * dtheta
        S = self.curve * self.S + (1 - self.curve) * np.square(dtheta)
        v = np.max([self.S, self.v], axis=0)

        dtheta[dtheta == 0] = 1
        M = M / dtheta

        ss = (self.stepsize / np.sqrt(v)) * M

        # if the denominator has a 0, then we end up with massive spikes in stepsize
        # if those spikes occur where the update is also 0, then we aren't actually making a huge update
        # so to meaningfully visualize the stepsize, we can get rid of those spikes by making the stepsize 0 in those cases
        # which is safe because the update was also 0
        ss = np.where((S == 0) & (dtheta == 0), np.zeros_like(ss), ss)

        return np.mean(ss, axis=1)

    def update(self, x, a, xp, r, gamma, p):
        experience = (x, a, xp, r, gamma, p)
        self.buffer.add(experience)

        dtheta = np.array(self.computeGradient(*experience))
        self.theta = self.theta + self.stepsize * dtheta

        self.M = self.momentum * self.M + (1 - self.momentum) * dtheta
        self.S = self.curve * self.S + (1 - self.curve) * np.square(dtheta)
        self.v = np.max([self.S, self.v], axis=0)
        self.theta = self.theta + (self.stepsize / np.sqrt(self.v)) * self.M

        for i in range(self.replay):
            experience = self.buffer.sample()[0]
            dtheta = np.array(self.computeGradient(*experience))

            self.M = self.momentum * self.M + (1 - self.momentum) * dtheta
            self.S = self.curve * self.S + (1 - self.curve) * np.square(dtheta)
            self.v = np.max([self.S, self.v], axis=0)
            self.theta = self.theta + (self.stepsize / np.sqrt(self.v)) * self.M

        self.last_p = p
        self.last_gamma = gamma
        self.dtheta = dtheta
