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

    def tde(self, r, gamma, v_tp1, v_t, p):
        raise NotImplementedError()

    def computeGradient(self, obs_t, a_t, obs_tp1, r, gamma, p):
        w = self.theta[0]
        v_tp1 = np.dot(obs_tp1, w)
        v_t = np.dot(obs_t, w)

        delta = self.tde(r, gamma, v_tp1, v_t, p)

        dw = delta * obs_t
        dh = np.zeros(self.features)
        return [ dw, dh ]

    def update(self, obs_t, a_t, obs_tp1, r, gamma, p):
        dtheta = self.computeGradient(obs_t, a_t, obs_tp1, r, gamma, p)

        self.theta = self.theta + self.stepsize * dtheta

        self.last_p = p
        self.last_gamma = gamma

    def computeVarianceOfUpdates(self, experiences):
        updates = map(lambda ex: self.computeGradient(*ex), experiences)

        return np.mean(np.std(list(updates), axis=0, ddof=1), axis=1)

    def computeVarianceOfTDE(self, experiences):
        def computeTDE(experience):
            obs_t, a_t, obs_tp1, r, gamma, p = experience
            w = self.theta[0]
            v_tp1 = np.dot(obs_tp1, w)
            v_t = np.dot(obs_t, w)

            return self.tde(r, gamma, v_tp1, v_t, p)

        tdes = map(computeTDE, experiences)
        return np.std(list(tdes), axis=0, ddof=1)

    def reset(self):
        pass

    def value(self, obs):
        w = self.theta[0]
        return np.dot(obs, w)
