import numpy as np
from .BaseTD import BaseTD

class LSTD(BaseTD):
    def __init__(self, features, params):
        super().__init__(features, {'gamma': 0.0, 'alpha': 0.0, 'alpha_h': 0.0})
