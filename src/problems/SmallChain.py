import numpy as np
from PyFixedReps.BaseRepresentation import BaseRepresentation
from src.problems.Chain import BaseChain, Policy5050, Policy4060, RepDependent, RepInverted, RepTabular
from src.environments.Chain import Chain as ChainEnv
from src.utils.rlglue import OffPolicyWrapper
from src.utils.policies import Policy

class SmallChain(BaseChain):
    def _getSize(self):
        return 5

    def _getdb(self):
        return np.array([0.111111, 0.222222, 0.333333, 0.222222, 0.111111, 0])

    def getSteps(self):
        return 1000

# ---------------
# -- Resultant --
# ---------------

class SmallChainDependent5050(Policy5050, RepDependent, SmallChain):
    pass

class SmallChainInverted5050(Policy5050, RepInverted, SmallChain):
    pass

class SmallChainTabular5050(Policy5050, RepTabular, SmallChain):
    pass

class SmallChainDependent4060(Policy4060, RepDependent, SmallChain):
    pass

class SmallChainInverted4060(Policy4060, RepInverted, SmallChain):
    pass

class SmallChainTabular4060(Policy4060, RepTabular, SmallChain):
    pass
