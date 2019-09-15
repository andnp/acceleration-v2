import numpy as np
from src.problems.MediumChain import MediumChain, MediumChain5050
from PyFixedReps.TileCoder import TileCoder

# ======== Tile coder ======

class ScaledTC(TileCoder):
    def encode(self, s, a = None):
        return super().encode(s / 8, a)

class TCChain(MediumChain):
    def _getRepresentation(self, n, perm):
        params = perm['metaParameters']
        return ScaledTC({
            'dims': 1,
            'tilings': params['tilings'],
            'tiles': params['tiles'],
        })

# ---------------
# -- Resultant --
# ---------------

class MediumChainTC5050(MediumChain5050, TCChain):
    pass
