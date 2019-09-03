from src.problems.BairdCounterexample import BairdCounterexample
from src.problems.BairdRMSPBE import BairdRMSPBE
from src.problems.Boyan import Boyan
from src.problems.Chain import Chain2575, Chain4060

def getProblem(name):
    if name == 'Baird':
        return BairdCounterexample

    if name == 'Boyan':
        return Boyan

    if name == 'Chain2575':
        return Chain2575

    if name == 'Chain4060':
        return Chain4060

    if name == 'BairdRMSPBE':
    	return BairdRMSPBE

    raise NotImplementedError()
