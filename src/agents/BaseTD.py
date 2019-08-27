import numpy as np

class BaseTD:
    def __init__(self, features, params):
        self.features = features
        self.gamma = params['gamma']
        self.lambdaa = params['lambda']
        self.alpha = params['alpha']
        self.alpha_h = params.get('alpha_h')
        if self.alpha_h is None:
            self.alpha_h = params['ratio'] * self.alpha

        self.z = np.zeros(features)
        self.theta = np.zeros((2, features))
        self.last_p = 1
        self.last_gamma = 1
        self.stepsize = np.tile([self.alpha, self.alpha_h], (features, 1)).T

    def tde(self, r, gamma, v_tp1, v_t, p):
        raise NotImplementedError()

    def nextTrace(self, obs_t):
        return self.last_p * self.last_gamma * self.lambdaa * self.z + obs_t

    def computeGradient(self, obs_t, a_t, obs_tp1, r, gamma, p):
        w = self.theta[0]
        v_tp1 = np.dot(obs_tp1, w)
        v_t = np.dot(obs_t, w)

        delta = self.tde(r, gamma, v_tp1, v_t, p)

        dw = delta * self.z
        dh = np.zeros(self.features)
        return [ dw, dh ]

    def update(self, obs_t, a_t, obs_tp1, r, gamma, p):
        self.z = self.nextTrace(obs_t)

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
        self.z = np.zeros(self.features)

    def value(self, obs):
        w = self.theta[0]
        return np.dot(obs, w)

# TDE
def correctWholeTDE(r, gamma, v_tp1, v_t, p):
    return p * (r + gamma * v_tp1 - v_t)

def correctTDTarget(r, gamma, v_tp1, v_t, p):
    return p * (r + gamma * v_tp1) - v_t

# This version is highly biased because the reward is distributed according to the behavior policy
# the reward needs to be corrected as well for the predition setting.
def correctBootstrap(r, gamma, v_tp1, v_t, p):
    return (r + gamma * p * v_tp1) - v_t

# first:  correct tde_w

def getTDMethod(code):
    tde = [ correctWholeTDE, correctTDTarget, correctBootstrap ][int(code[0])]

    class TDMethod(BaseTD):
        def tde(self, r, gamma, v_tp1, v_t, p):
            return tde(r, gamma, v_tp1, v_t, p)

    return TDMethod
