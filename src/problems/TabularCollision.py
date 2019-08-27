from src.problems.StandardCollision import StandardCollision
from src.representations.BinaryEncoder import BinaryEncoder

class TabularCollision(StandardCollision):
    def _buildRepresentation(self):
        return BinaryEncoder(active=1, bits=8, states=8)
