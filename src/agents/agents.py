from .LSTD import LSTD
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
from .htd.HTD import HTD
from .htd.HTDadagrad import HTDadagrad
from .htd.HTDschedule import HTDschedule
from .smooth_tdc.SmoothTDC import SmoothTDC
from .smooth_tdc.SmoothTDCschedule import SmoothTDCschedule
from .smooth_tdc.SmoothTDCadagrad import SmoothTDCadagrad
from .regh_tdc.ReghTDC import ReghTDC
from .regh_tdc.ReghTDCadagrad import ReghTDCadagrad
from .regh_tdc.ReghTDCschedule import ReghTDCschedule
from .regh_gtd2.ReghGTD2 import ReghGTD2
from .regh_gtd2.ReghGTD2adagrad import ReghGTD2adagrad
from .regh_gtd2.ReghGTD2schedule import ReghGTD2schedule

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

    elif name == 'SmoothTDC':
        return SmoothTDC

    elif name == 'SmoothTDCadagrad':
        return SmoothTDCadagrad

    elif name == 'SmoothTDCschedule':
        return SmoothTDCschedule

    elif name == 'LSTD':
        return LSTD

    elif name == 'HTD':
        return HTD

    elif name == 'HTDadagrad':
        return HTDadagrad

    elif name == 'HTDschedule':
        return HTDschedule

    elif name == 'ReghTDC':
        return ReghTDC

    elif name == 'ReghTDCadagrad':
        return ReghTDCadagrad

    elif name == 'ReghTDCschedule':
        return ReghTDCschedule

    elif name == 'ReghGTD2':
        return ReghGTD2

    elif name == 'ReghGTD2adagrad':
        return ReghGTD2adagrad

    elif name == 'ReghGTD2schedule':
        return ReghGTD2schedule

    raise Exception('Unexpected agent given')
