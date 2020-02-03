import numpy as np
from src.agents.BaseTD import BaseTD
from .LinearKernel import LinearKernel

class GaussianKernel(LinearKernel):
    def __init__(self, features, params):
        super().__init__(features, params)
        self.bandwidth = params.get('bandwidth', 0.5)
        print(self.alpha, self.bandwidth, self.params['batch_size'])

    def kernel(self, x1, x2):
        diff = x1 - x2
        squared = np.sum(np.square(diff))
        return np.exp(-1 * squared / np.square(self.bandwidth))


class GaussianKernelAdagrad(GaussianKernel):
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
