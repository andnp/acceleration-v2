from src.problems.StandardCollision import StandardCollision
from src.representations.SparseRandomLinear import SparseRandomLinear

class RandomLinearCollision(StandardCollision):
    def _buildRepresentation(self):
        return SparseRandomLinear(8, 100, 0.9)
