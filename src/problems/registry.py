from src.problems.BairdCounterexample import BairdCounterexample
from src.problems.Boyan import Boyan

def getProblem(name):
    if name == 'Baird':
        return BairdCounterexample

    if name == 'Boyan':
        return Boyan

    raise NotImplementedError()
