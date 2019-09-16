from .BaseTD import BaseTD
from .optimizers.AdaGrad import AdaGrad
from .optimizers.Schedule import Schedule
from .optimizers.AMSGrad import AMSGrad
from .tdc.TDC import TDC
from .tdc.TDCadagrad import TDCadagrad
from .tdc.TDCsecondaryAdagrad import TDCsecondaryAdagrad
from .tdc.TDCAdaGradPNorm import TDCAdaGradPNorm
from .tdc.TDCschedule import TDCschedule
from .tdc.TDCamsgrad import TDCamsgrad
from .gtd2.GTD2 import GTD2
from .gtd2.GTD2adagrad import GTD2adagrad
from .gtd2.GTD2secondaryAdagrad import GTD2secondaryAdagrad
from .gtd2.GTD2AdaGradPNorm import GTD2AdaGradPNorm
from .gtd2.GTD2schedule import GTD2schedule
from .gtd2.GTD2amsgrad import GTD2amsgrad
from .etdc.ETDC import ETDC
from .etdc.ETDCadagrad import ETDCadagrad
from .etdc.ETD import ETD
from .etdc.ETDadagrad import ETDadagrad

def getAgent(name):
    if name == 'TDC':
        return TDC

    elif name == 'GTD2':
        return GTD2

    elif name == 'GTD2adagrad':
        return GTD2adagrad

    elif name == 'GTD2secondaryAdagrad':
        return GTD2secondaryAdagrad

    elif name == 'GTD2schedule':
        return GTD2schedule

    elif name == 'GTD2amsgrad':
        return GTD2amsgrad

    elif name == 'TDCsecondaryAdagrad':
        return TDCsecondaryAdagrad

    elif name == 'TDCadagrad':
        return TDCadagrad

    elif name == 'TD':
        return BaseTD

    elif name == 'TDadagrad':
        return AdaGrad

    elif name == 'TDschedule':
        return Schedule

    elif name == 'TDamsgrad':
        return AMSGrad

    elif name == 'TDCAdaGradPNorm':
        return TDCAdaGradPNorm

    elif name == 'TDCschedule':
        return TDCschedule

    elif name == 'TDCamsgrad':
        return TDCamsgrad

    elif name == 'GTD2AdaGradPNorm':
        return GTD2AdaGradPNorm

    elif name == 'ETDC':
        return ETDC

    elif name == 'ETDCadagrad':
        return ETDCadagrad

    elif name == 'ETDadagrad':
        return ETDadagrad

    elif name == 'ETD':
        return ETD

    raise Exception('Unexpected agent given')
