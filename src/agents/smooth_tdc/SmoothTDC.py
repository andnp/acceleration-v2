import numpy as np
from src.agents.BaseTD import BaseTD
from src.utils.buffers import CircularBuffer

class SmoothTDC(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)

        self.use_ideal_h = params.get('use_ideal_h', False)
        self.average = params['averageType']

        if self.average == 'buffer':
            self.buffer_size = params['buffer']
            self.buffer = CircularBuffer(self.buffer_size)
        elif self.average == 'ema':
            self.smoothing = params['smoothing']
            self.A = 0
        else:
            raise NotImplementedError('Expected "average" to be either "buffer" or "ema"')

    def _emaDelta(self, x):
        w, h = self.theta
        delta_hat = h.dot(x)
        self.A = self.smoothing * self.A + (1 - self.smoothing) * delta_hat
        return self.A

    def _bufferDelta(self, x):
        w, h = self.theta
        delta_hat = h.dot(x)
        self.buffer.add(delta_hat)
        return np.mean(self.buffer.getAll())

    def _getDeltaHat(self, x):
        if self.average == 'buffer':
            return self._bufferDelta(x)
        elif self.average == 'ema':
            return self._emaDelta(x)
        else:
            raise Exception()

    def computeGradient(self, x, a, xp, r, gamma, p):
        w, h = self.theta
        vp = w.dot(xp)
        v = w.dot(x)

        if self.use_ideal_h:
            h = self.getIdealH()

        delta = r + gamma * vp - v
        delta_hat = self._getDeltaHat(x)

        dw = p * (delta * x - gamma * delta_hat * xp)
        dh = (p * delta - delta_hat) * x

        return [dw, dh]
