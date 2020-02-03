import numpy as np
from src.agents.BaseTD import BaseTD

class LinearKernel(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, params)

        self.use_ideal_h = params.get('use_ideal_h', False)

    def kernel(self, x1, x2):
        return x1.dot(x2)

    def _compute_update(self, gen):
        w, _ = self.theta

        ex1 = gen.sample(samples=1)[0]
        ex2 = gen.sample(samples=1)[0]

        x1, a1, x1p, r1, g1, p1 = ex1
        x2, a2, x2p, r2, g2, p2 = ex2

        k = self.kernel(x1, x2)
        v1 = w.dot(x1)
        v1p = w.dot(x1p)

        delta1 = r1 + g1 * v1p - v1
        ddelta2 = x2 - g2 * x2p

        return k * delta1 * ddelta2

    def batch_update(self, gen):
        num = self.params['batch_size']
        grads = np.zeros(self.theta.shape)
        for _ in range(num):
            grads[0] += self._compute_update(gen)

        grad = grads / num

        self.theta = self.theta + self.stepsize * grad

class LinearKernelAdagrad(LinearKernel):
    def __init__(self, features, params):
        super().__init__(features, params)
        self.S = np.zeros((2, features))

    def batch_update(self, gen):
        w, _ = self.theta
        num = self.params['batch_size']
        grads = np.zeros(self.theta.shape)
        for _ in range(num):
            grads[0] += self._compute_update(gen)

        grad = grads / num

        self.S = self.S + np.square(grad)
        self.theta = self.theta + (self.stepsize / (np.sqrt(self.S) + 1e-8)) * grad
