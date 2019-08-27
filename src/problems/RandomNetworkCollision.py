from src.problems.StandardCollision import StandardCollision
from src.representations.SparseNetwork import SparseNetwork

class RandomNetworkCollision(StandardCollision):
    def _buildRepresentation(self):
        return SparseNetwork(8, [2000, 25], 0.95)
