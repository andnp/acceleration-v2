import numpy as np
from src.utils.buffers import CircularBuffer

class BaseTD:
    def __init__(self, features, params):
        self.params = params
        self.features = features
        self.gamma = params['gamma']
        self.alpha = params['alpha']

        self.alpha_h = params.get('alpha_h')
        if self.alpha_h is None:
            self.alpha_h = params.get('ratio', 0) * self.alpha

        self.h_variance = params.get('h_variance', 0)
        self.replay = params.get('replay', 0)
        self.buffer_size = params.get('buffer_size', 1)
        self.buffer = CircularBuffer(self.buffer_size)

        self.theta = np.zeros((2, features))

        if self.h_variance > 0:
            self.theta[1] = np.random.normal(0, self.h_variance, size=features)

        self.last_p = 0
        self.last_gamma = 1
        self.stepsize = np.tile([self.alpha, self.alpha_h], (features, 1)).T

        # tracking variables - these are exposed only for data collection
        self.ideal_h_params = (None, None, None)
        self.dtheta = None

    def computeGradient(self, obs_t, a_t, obs_tp1, r, gamma, p):
        w = self.theta[0]
        v_tp1 = np.dot(obs_tp1, w)
        v_t = np.dot(obs_t, w)

        delta = p * (r + gamma * v_tp1 - v_t)

        dw = delta * obs_t
        dh = np.zeros(self.features)
        return [ dw, dh ]

    def update(self, x, a, xp, r, gamma, p):
        experience = (x, a, xp, r, gamma, p)
        self.buffer.add(experience)

        dtheta = self.computeGradient(*experience)
        self.theta = self.theta + self.stepsize * dtheta

        for i in range(self.replay):
            experience = self.buffer.sample()[0]
            dtheta = self.computeGradient(*experience)
            self.theta = self.theta + self.stepsize * dtheta

        self.last_p = p
        self.last_gamma = gamma
        self.dtheta = dtheta

    def batch_update(self, gen):
        num = self.params['batch_size']
        exps = gen.sample(samples=num)
        shape = (num,) + self.theta.shape
        grads = np.zeros(shape)
        for i in range(num):
            grads[i] = self.computeGradient(*exps[i])

        grad = np.mean(grads, axis=0)

        self.theta = self.theta + self.stepsize * grad

    def reset(self):
        pass

    def value(self, obs):
        w = self.theta[0]
        return np.dot(obs, w)

    # idealH
    def getIdealH(self):
        A, b, C = self.ideal_h_params
        w, _ = self.theta
        s = np.linalg.lstsq(C, -A.dot(w) + b, rcond=None)
        h = s[0]
        # h = Cinv.dot(-A.dot(w) + b)
        return h

    # variance helpers
    def computeVarianceOfUpdates(self, experiences):
        updates = map(lambda ex: self.computeGradient(*ex), experiences)

        return np.mean(np.std(list(updates), axis=0, ddof=1), axis=1)

    # expectation helpers
    def computeMeanOfUpdates(self, experiences):
        updates = map(lambda ex: self.computeGradient(*ex), experiences)

        return np.mean(np.mean(list(updates), axis=0), axis=1)

    # effective stepsize helpers
    def _stepsize(self):
        return [self.alpha, self.alpha_h]
