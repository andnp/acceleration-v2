from .BaseTD import BaseTD
from .TDC import TDC
from .GTD2 import GTD2
from .GTD2adagrad import GTD2adagrad
from .GTD2secondaryAdagrad import GTD2secondaryAdagrad
from .TDCadagrad import TDCadagrad
from .AdaGrad import AdaGrad
from .TDCsecondaryAdagrad import TDCsecondaryAdagrad
from .GTD2AdaGradPNorm import GTD2AdaGradPNorm
from .TDCAdaGradPNorm import TDCAdaGradPNorm

def getAgent(name):
    if name == 'TDC':
        return TDC

    elif name == 'GTD2':
        return GTD2

    elif name == 'GTD2adagrad':
        return GTD2adagrad

    elif name == 'GTD2secondaryAdagrad':
        return GTD2secondaryAdagrad

    elif name == 'TDCsecondaryAdagrad':
        return TDCsecondaryAdagrad

    elif name == 'TDCadagrad':
        return TDCadagrad

    elif name == 'TD':
        return BaseTD

    elif name == 'TDadagrad':
        return AdaGrad
    elif name == 'TDCAdaGradPNorm':
        return TDCAdaGradPNorm

    elif name == 'GTD2AdaGradPNorm':
        return GTD2AdaGradPNorm

    raise Exception('Unexpected agent given')
