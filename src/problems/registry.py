from src.problems.BairdCounterexample import BairdCounterexample
from src.problems.Boyan import Boyan
from src.problems.Chain import Chain2575, Chain4060, Chain5050
from src.problems.MediumChain import MediumChainTabular5050
from src.problems.MediumChainTC import MediumChainTC5050
from src.problems.SmallChain import SmallChainTabular5050, SmallChainInverted5050, SmallChainDependent5050
from src.problems.SmallChainLeftZero import SmallChainTabular5050LeftZero, SmallChainInverted5050LeftZero, SmallChainDependent5050LeftZero

def getProblem(name):
    if name == 'Baird':
        return BairdCounterexample

    if name == 'Boyan':
        return Boyan

    if name == 'Chain2575':
        return Chain2575

    if name == 'Chain4060':
        return Chain4060

    if name == 'Chain5050':
        return Chain5050

    if name == 'MediumChainTabular5050':
        return MediumChainTabular5050

    if name == 'MediumChainTC5050':
        return MediumChainTC5050

    if name == 'SmallChainTabular5050':
        return SmallChainTabular5050

    if name == 'SmallChainDependent5050':
        return SmallChainDependent5050

    if name == 'SmallChainInverted5050':
        return SmallChainInverted5050

    if name == 'SmallChainDependent5050LeftZero':
        return SmallChainDependent5050LeftZero

    if name == 'SmallChainInverted5050LeftZero':
        return SmallChainInverted5050LeftZero

    if name == 'SmallChainTabular5050LeftZero':
        return SmallChainTabular5050LeftZero

    raise NotImplementedError()


