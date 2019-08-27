from src.problems.StandardCollision import StandardCollision
from src.problems.TabularCollision import TabularCollision
from src.problems.CollisionConvergence import CollisionConvergence
from src.problems.RandomLinearCollision import RandomLinearCollision
from src.problems.RandomNetworkCollision import RandomNetworkCollision
from src.problems.BigRhoCollision import BigRhoCollision
from src.problems.BairdCounterexample import BairdCounterexample
from src.problems.BairdConvergence import BairdConvergence

def getProblem(name):
    if name == 'StandardCollision':
        return StandardCollision

    if name == 'TabularCollision':
        return TabularCollision

    if name == 'CollisionConvergence':
        return CollisionConvergence

    if name == 'RandomLinearCollision':
        return RandomLinearCollision

    if name == 'RandomNetworkCollision':
        return RandomNetworkCollision

    if name == 'BigRhoCollision':
        return BigRhoCollision

    if name == 'Baird':
        return BairdCounterexample

    if name == 'BairdConvergence':
        return BairdConvergence

    raise NotImplementedError()
