import numpy as np
from src.problems.TabularCollision import TabularCollision
import src.utils.policies as Policy

class BigRhoCollision(TabularCollision):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.behavior = Policy.fromStateArray(
            [[1.0, 0.0]] * 4 +
            [[0.2, 0.8]] * 4
        )

        self.db = getStateDistribution()

    def getSteps(self):
        return 40000

def getStateDistribution():
    # attempt to load db
    # if not found, assume we will create it first
    # TODO: generate if not found
    try:
        return np.genfromtxt(f'baselines/BigRhoCollision_db.csv')
    except Exception as e:
        print('db not found, must create it before running experiments')
        raise e
