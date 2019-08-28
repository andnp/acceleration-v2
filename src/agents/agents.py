from .TDC import TDC
from .GTD2 import GTD2

def getAgent(name):
    if name == 'TDC':
        return TDC

    elif name == 'GTD2':
        return GTD2

    raise Exception('Unexpected agent given')
