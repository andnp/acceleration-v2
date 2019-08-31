from .BaseTD import BaseTD
from .TDC import TDC
from .GTD2 import GTD2
from .GTD2adagrad import GTD2adagrad
from .TDCadagrad import TDCadagrad

def getAgent(name):
    if name == 'TDC':
        return TDC

    elif name == 'GTD2':
        return GTD2

    elif name == 'GTD2adagrad':
        return GTD2adagrad

    elif name == 'TDCadagrad':
        return TDCadagrad

    elif name == 'TD':
        return BaseTD

    raise Exception('Unexpected agent given')
