import numpy as np
from src.agents.BaseTD import BaseTD
from src.utils.buffers import CircularBuffer

class SmoothTDC(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)

        self.use_ideal_h = params.get('use_ideal_h', False)
        self.average = params['averageType']

        self.update_buffer = False

        self.buffer_size = params['buffer']
        if self.average == 'buffer':
            self.buffer = CircularBuffer(self.buffer_size)
        elif self.average == 'window':
            self.buffer = CircularBuffer(self.buffer_size)
        elif self.average == 'ema':
            self.smoothing = 1 - 1/self.buffer_size
            self.A = 0
            self.o = 0
        elif self.average == 'ema_x':
            self.smoothing = 1 - 1/self.buffer_size
            self.A = np.zeros(features)
            self.o = 0
        else:
            raise NotImplementedError('Expected "average" to be either "buffer" or "ema"')

    def _emaDelta(self, delta):
        o = self.o + self.smoothing * (1 - self.o)
        ss = self.smoothing / o
        A = ss * self.A + (1 - ss) * delta

        if self.update_buffer:
            self.A = A
            self.o = o

        return A

    def _bufferDelta(self, delta):
        if self.update_buffer:
            self.buffer.add(delta)

        return np.mean(self.buffer.getAll())

    def _windowDelta(self, x, xp, r, gamma, p):
        if self.update_buffer:
            self.buffer.add((x, xp, r, gamma, p))

        w, h = self.theta
        deltas = list(map(lambda ex: ex[4] * (ex[2] + ex[3] * w.dot(ex[1]) - w.dot(ex[0])), self.buffer.getAll()))
        return np.mean(deltas)

    def _smoothDelta(self, delta, x, xp, r, gamma, p):
        if self.average == 'buffer':
            return self._bufferDelta(p * delta) * x
        elif self.average == 'window':
            return self._windowDelta(x, xp, r, gamma, p) * x
        elif self.average == 'ema':
            return self._emaDelta(p * delta) * x
        elif self.average == 'ema_x':
            return self._emaDelta(p * delta * x)
        else:
            raise Exception()

    def computeGradient(self, x, a, xp, r, gamma, p):
        w, h = self.theta
        vp = w.dot(xp)
        v = w.dot(x)

        if self.use_ideal_h:
            h = self.getIdealH()

        exp_delta_x = self._smoothDelta(r + gamma * vp - v, x, xp, r, gamma, p)

        delta = r + gamma * vp - v
        delta_hat = h.dot(x)

        dw = (exp_delta_x - p * gamma * delta_hat * xp)
        dh = (p * delta - delta_hat) * x

        return [dw, dh]

    def update(self, obs_t, a_t, obs_tp1, r, gamma, p):
        self.update_buffer = True
        dtheta = self.computeGradient(obs_t, a_t, obs_tp1, r, gamma, p)
        self.update_buffer = False

        # print(self.stepsize / (np.sqrt(self.S) + 1e-8))
        self.theta = self.theta + self.stepsize * dtheta

        self.last_p = p
        self.last_gamma = gamma
        self.dtheta = dtheta
