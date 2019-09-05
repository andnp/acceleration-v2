from src.problems.Chain import BaseChain
from src.utils.policies import Policy
from PyFixedReps.TileCoder import TileCoder

class ScaledTC(TileCoder):
    def encode(self, s, a = None):
        return super().encode(s / 18, a)

class TCChain(BaseChain):
    def _getRepresentation(self, n, perm):
        params = perm['metaParameters']
        return ScaledTC({
            'dims': 1,
            'tilings': params['tilings'],
            'tiles': params['tiles'],
        })

    def getTarget(self, N):
        return Policy(lambda s: [0.5, 0.5])
