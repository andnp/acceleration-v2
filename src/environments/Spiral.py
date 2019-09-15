import numpy as np
from RlGlue import BaseEnvironment

# Constants
N = 3

class Spiral(BaseEnvironment):
    def __init__(self):
        self.state=0

    def start(self):
        self.state = np.random.randint(N)
        return self.state

    def step(self, a):
        t = 1 if np.random.rand()>0.5 else 0
        self.state = (self.state+t) % N
        return (0, self.state, False)
