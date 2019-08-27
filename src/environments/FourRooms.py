import numpy as np
from RlGlue import BaseEnvironment

# Constants
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

WALLS = np.array([[0, 5], [2, 5], [3, 5], [4, 5], [5, 5], [6, 4], [7, 4], [9, 4], [10, 4], [5, 0], [5, 2], [5, 3], [5, 4], [5, 6], [5, 7], [5, 9], [5, 10]])

PROB_RANDOM_ACT = 1/3

def isRowInArray(arr, row):
    return (arr == row).all(-1).any()

def isWall(state):
    return np.any(state < 0) or np.any(state > 1) or isRowInArray(WALLS, state)

def takeAction(state, act):
    next_state = state.copy()
    if act == UP:
        next_state[1] += 1
    elif act == RIGHT:
        next_state[0] += 1
    elif act == DOWN:
        next_state[1] -= 1
    elif act == LEFT:
        next_state[0] -= 1

    if isWall(next_state):
        return state

    return next_state

class FourRooms(BaseEnvironment):
    def __init__(self):
        self.state = np.array([0, 0])

    def start(self):
        self.state = np.array([0, 0])
        return self.state

    def step(self, a):
        if np.random.random() < PROB_RANDOM_ACT:
            next_state = takeAction(self.state, np.random.randint(4))
        else:
            next_state = takeAction(self.state, a)

        self.state = next_state

        return (0, self.state, False)
