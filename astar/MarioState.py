from enum import Enum

# Extra space added at the start and end of levels (LevelParser.BUFFER_WIDTH)
BUFFER_WIDTH = 15

LENIENCY_TILES = {
    0: 0.0,    # solid
    1: 0.0,    # breakable
    2: 0.0,    # passable
    3: 1.0,    # question with coin
    4: 1.0,    # question with power up
    5: 0.0,    # coin
    6: -0.5,   # tube
    7: -0.5,   # piranha plant tube
    8: -0.5,   # bullet bill
    9: -1.0,   # goomba
    10: -1.0,  # green koopas + paratroopas
    11: -1.0,  # red koopas + paratroopas
    12: -1.0,  # spiny + winged spiny
}


NEGATIVE_SPACE_TILES = {
    0: 1,   # solid
    1: 1,   # breakable
    2: 0,   # passable
    3: 1,   # question with coin
    4: 1,   # question with power up
    5: 0,   # coin
    6: 1,   # tube
    7: 1,   # piranha plant tube
    8: 1,   # bullet bill
    9: 0,   # goomba
    10: 0,  # green koopas + paratroopas
    11: 0,  # red koopas + paratroopas
    12: 0,  # spiny + winged spiny
}


class MarioState:

    def __init__(self, level, jumpVelocity, x, y):
        self.level = level # [[int]] 2d array of tile types
        self.jumpVelocity = jumpVelocity
        self.x = x
        self.y = y

    @classmethod
    def from_level(cls, level):
        """Default constructor: start at top-left, two tiles above the floor."""
        return cls(level, 0, 0, 2)

    # simple greedy heuristic: horizontal distance from Mario to the right edge (flagpole)
    def moveRight(self):
        return len(self.level[0]) - self.x

    class MarioAction:

        def __init__(self, direction):
            self.direction = direction

        class DIRECTION(Enum):
            JUMP = 0
            LEFT = 1
            RIGHT = 2

        def getDirection(self):
            return self.direction

        def __eq__(self, other):
            if isinstance(other, MarioState.MarioAction):
                return other.direction == self.direction
            return False

        def __hash__(self):
            return hash(self.direction)

        def __str__(self):
            return str(self.direction)

    @staticmethod
    def preProcessLevel(level):
        """
        Add opening/closing buffer spaces to the level, fixes pipes and bullet bills
        """
        extraStones = BUFFER_WIDTH
        height = len(level)
        width = len(level[0])
        tmpLevel = []

        for i in range(height):
            tile = 2
            if i == height - 1:
                tile = 0
            row = [tile] * extraStones
            row.extend(level[i])
            row.extend([tile] * extraStones)
            tmpLevel.append(row)

        for y in range(height - 1, -1, -1):
            for x in range(width - 1, -1, -1):
                tile = level[y][x]
                if tile == 8 and (y + 1 < height and level[y + 1][x] == 2):
                    MarioState.setTileAtPosition(tmpLevel, x + extraStones, y + 1, tile)
                    for i in range(y + 2, height):
                        if level[i][x] == 2:
                            MarioState.setTileAtPosition(tmpLevel, x + extraStones, i, tile)
                        else:
                            break
                if (tile == 6 or tile == 7) and (y + 1 < height and level[y + 1][x] == 2):
                    MarioState.setTileAtPosition(tmpLevel, x + extraStones + 1, y, tile)
                    MarioState.setTileAtPosition(tmpLevel, x + extraStones, y + 1, tile)
                    MarioState.setTileAtPosition(tmpLevel, x + extraStones + 1, y + 1, tile)
                    for i in range(y + 2, height):
                        if level[i][x] == 2:
                            MarioState.setTileAtPosition(tmpLevel, x + extraStones, i, tile)
                            MarioState.setTileAtPosition(tmpLevel, x + extraStones + 1, i, tile)
                        else:
                            break
        return tmpLevel

    def tileAtPosition(self, x, y):
        return self.level[y][x]

    @staticmethod
    def setTileAtPosition(level, x, y, tile):
        level[y][x] = tile

    def inBounds(self, x, y):
        return 0 <= y and y < len(self.level) and 0 <= x and x < len(self.level[0])

    def isGoalX(self, x):
        return x == len(self.level[0]) - 1

    def passable(self, x, y):
        if not self.inBounds(x, y):
            return False
        tile = self.tileAtPosition(x, y)
        return NEGATIVE_SPACE_TILES.get(tile) == 0 and LENIENCY_TILES.get(tile) == 0

    def get_successor(self, action):
        """Return the next state, or None if the action is not legal here."""
        newJumpVelocity = self.jumpVelocity
        newMarioX = self.x
        newMarioY = self.y

        # Falling off bottom of screen 
        if not self.inBounds(self.x, self.y + 1):
            return None

        if newJumpVelocity == 0:  # Not mid-jump
            if self.passable(newMarioX, newMarioY + 1):  # Falling
                newMarioY += 1  # Fall down
            elif action.getDirection() == self.MarioAction.DIRECTION.JUMP:  # Start jump
                newJumpVelocity = 5  # Accelerate up
        elif action.getDirection() == self.MarioAction.DIRECTION.JUMP:
            return None  # Can't jump mid-jump. Reduces search space.

        if newJumpVelocity > 0:  # Jumping up
            if self.passable(newMarioX, newMarioY - 1):
                newMarioY -= 1       # Jump up
                newJumpVelocity -= 1  # decelerate
            else:
                newJumpVelocity = 0  # Can't jump if blocked above
            # TODO: Add breakable case

        # Right movement
        if action.getDirection() == self.MarioAction.DIRECTION.RIGHT:
            if self.passable(newMarioX + 1, newMarioY):
                newMarioX += 1
            elif self.y == newMarioY:  # vertical position did not change
                # No new state: not jumping/falling and could not move right
                return None

        # Left movement
        if action.getDirection() == self.MarioAction.DIRECTION.LEFT:
            if self.passable(newMarioX - 1, newMarioY):
                newMarioX -= 1
            elif self.y == newMarioY:  # vertical position did not change
                # No new state: not jumping/falling and could not move left
                return None

        if not self.inBounds(newMarioX, newMarioY):
            return None

        return MarioState(self.level, newJumpVelocity, newMarioX, newMarioY)

    def get_successors(self):
        """List of (next_state, action, step_cost) reachable from current state"""
        successors = []
        for a in self.getLegalActions(self):
            successor = self.get_successor(a)
            successors.append((successor, a, self.stepCost()))
        return successors

    def getLegalActions(self, state):
        """A list of valid actions for playing Mario levels."""
        possible = []
        for direction in self.MarioAction.DIRECTION:
            if state.get_successor(self.MarioAction(direction)) is not None:
                possible.append(self.MarioAction(direction))
        return possible

    def isGoal(self):
        """True once Mario reaches the last x coordinate in the level."""
        return self.isGoalX(self.x)

    def stepCost(self):
        return 1

    def __hash__(self):
        prime = 31
        result = 1
        result = prime * result + self.jumpVelocity
        result = prime * result + self.x
        result = prime * result + self.y
        return result

    def __eq__(self, other):
        if self is other:
            return True
        if other is None:
            return False
        if not isinstance(other, MarioState):
            return False
        return (self.jumpVelocity == other.jumpVelocity
                and self.x == other.x
                and self.y == other.y)

    def __str__(self):
        return f"({self.x},{self.y})"
