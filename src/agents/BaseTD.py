import numpy as np

class BaseTD:
    def __init__(self, features, params):
        self.features = features
        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.alpha_h = params.get('alpha_h')
        if self.alpha_h is None:
            self.alpha_h = params['ratio'] * self.alpha

        self.theta = np.zeros((2, features))
        self.last_p = 1
        self.last_gamma = 1
        self.stepsize = np.tile([self.alpha, self.alpha_h], (features, 1)).T
        self.dw = None

        self.ideal_h_params = (None, None, None)

    def computeGradient(self, obs_t, a_t, obs_tp1, r, gamma, p):
        w = self.theta[0]
        v_tp1 = np.dot(obs_tp1, w)
        v_t = np.dot(obs_t, w)

        delta = p * (r + gamma * v_tp1 - v_t)

        self.dw = delta * obs_t
        dh = np.zeros(self.features)
        return [ self.dw, dh ]

    def update(self, obs_t, a_t, obs_tp1, r, gamma, p):
        dtheta = self.computeGradient(obs_t, a_t, obs_tp1, r, gamma, p)

        # print(self.stepsize / (np.sqrt(self.S) + 1e-8))
        self.theta = self.theta + self.stepsize * dtheta

        self.last_p = p
        self.last_gamma = gamma

    def reset(self):
        pass

    def value(self, obs):
        w = self.theta[0]
        return np.dot(obs, w)

    # idealH
    def getIdealH(self):
        A, b, Cinv = self.ideal_h_params
        w, _ = self.theta
        h = Cinv.dot(-A.dot(w) + b)
        return h

    # variance helpers
    def computeVarianceOfUpdates(self, experiences):
        updates = map(lambda ex: self.computeGradient(*ex), experiences)

        return np.mean(np.std(list(updates), axis=0, ddof=1), axis=1)
