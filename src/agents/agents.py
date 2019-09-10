from .BaseTD import BaseTD
from .optimizers.AdaGrad import AdaGrad
from .tdc.TDC import TDC
from .tdc.TDCadagrad import TDCadagrad
from .tdc.TDCsecondaryAdagrad import TDCsecondaryAdagrad
from .tdc.TDCAdaGradPNorm import TDCAdaGradPNorm
from .gtd2.GTD2 import GTD2
from .gtd2.GTD2adagrad import GTD2adagrad
from .gtd2.GTD2secondaryAdagrad import GTD2secondaryAdagrad
from .gtd2.GTD2AdaGradPNorm import GTD2AdaGradPNorm
from .GTD3adagrad import GTD3
from .GTD4adagrad import GTD4
from .GTD5adagrad import GTD5

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

    elif name == 'GTD3adagrad':
        return GTD3
    elif name == 'GTD4adagrad':
        return GTD4
    elif name == 'GTD5adagrad':
        return GTD5

    raise Exception('Unexpected agent given')
