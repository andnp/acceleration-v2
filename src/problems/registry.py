from src.problems.BairdCounterexample import BairdCounterexample
from src.problems.Boyan import Boyan
from src.problems.Chain import Chain

def getProblem(name):
    if name == 'Baird':
        return BairdCounterexample

    if name == 'Boyan':
        return Boyan

    if name == 'Chain':
        return Chain

    raise NotImplementedError()
