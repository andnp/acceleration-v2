import numpy as np

# way faster than np.random.choice
def sample(arr):
    r = np.random.random()
    s = 0
    for i, p in enumerate(arr):
        s += p
        if s > r or s == 1:
            return i

    # worst case if we run into floating point error, just return the last element
    return len(arr) - 1

class Policy:
    def __init__(self, probs):
        self.probs = probs

    def selectAction(self, s):
        action_probabilities = self.probs(s)
        return sample(action_probabilities)

    def ratio(self, other, s, a):
        probs = self.probs(s)
        return probs[a] / other.probs(s)[a]

def fromStateArray(probs):
    return Policy(lambda s: probs[s])

def fromActionArray(probs):
    return Policy(lambda s: probs)
