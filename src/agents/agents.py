from .BaseTD import getTDMethod
from .TDC import getTDCMethod
from .GTD2 import getGTD2Method
from .GTD2_1 import GTD2_1
from .GTD3 import GTD3

def getAgent(name):
    # last three characters specify the method code
    code = name[-3:]

    if name == 'gtd2.1':
        return GTD2_1

    elif name == 'gtd3':
        return GTD3

    elif name.startswith('tdc'):
        return getTDCMethod(code)

    elif name.startswith('td'):
        code = name[-1:]
        return getTDMethod(code)

    elif name.startswith('gtd2'):
        return getGTD2Method(code)

    raise Exception('Unexpected agent given')
