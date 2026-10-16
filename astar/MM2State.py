from enum import Enum

# Extra space added at the start and end of levels (LevelParser.BUFFER_WIDTH)
BUFFER_WIDTH = 15

# Deadly static hazard (spikes, saws, ...). Kept distinct from solid so the agent can
# neither pass through it nor come to rest on top of it. Outside the 0-12 SMB id range
# so real Mario tiles are never mistaken for it.
HAZARD = 13

# Semisolid platforms, lifts and one-way walls. Passable from every side, but they
# stop a fall: you land on top and jump straight back up through.
PLATFORM = 14

# Doors and warp pipes. Solid like any wall, and reaching one ends the search since
# it leads out of the scene.
WARP = 15

# Springs. Stand on one like a platform; a jump off it clears walls your own cannot.
SPRING = 16

# Tiles a spring throws you, against 5 for a jump. Estimated, not measured.
SPRING_JUMP = 10

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
    13: 0.0,   # static hazard (spikes/saws)
    14: 0.0,   # semisolid platform
    15: 0.0,   # door / warp pipe
    16: 0.0,   # spring
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
    13: 1,  # static hazard occupies space (non-passable)
    14: 0,  # you can move through a platform, you just can't fall through it
    15: 1,  # a warp is a wall you step into, so it blocks like one
    16: 0,  # a spring is passable like a platform, and stops a fall the same way
}


class MM2State:

    def __init__(self, level, jumpVelocity, x, y):
        self.level = level # [[int]] 2d array of tile types
        self.jumpVelocity = jumpVelocity
        self.x = x
        self.y = y

    @classmethod
    def from_level(cls, level):
        """Default constructor: start at top-left, two tiles above the floor."""
        return cls(level, 0, 0, 2)

    # Greedy heuristic: horizontal distance to the right edge. Overshoots when the
    # nearest goal is a warp behind Mario. Costs expansions, not answers.
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
            if isinstance(other, MM2State.MarioAction):
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
                    MM2State.setTileAtPosition(tmpLevel, x + extraStones, y + 1, tile)
                    for i in range(y + 2, height):
                        if level[i][x] == 2:
                            MM2State.setTileAtPosition(tmpLevel, x + extraStones, i, tile)
                        else:
                            break
                if (tile == 6 or tile == 7) and (y + 1 < height and level[y + 1][x] == 2):
                    MM2State.setTileAtPosition(tmpLevel, x + extraStones + 1, y, tile)
                    MM2State.setTileAtPosition(tmpLevel, x + extraStones, y + 1, tile)
                    MM2State.setTileAtPosition(tmpLevel, x + extraStones + 1, y + 1, tile)
                    for i in range(y + 2, height):
                        if level[i][x] == 2:
                            MM2State.setTileAtPosition(tmpLevel, x + extraStones, i, tile)
                            MM2State.setTileAtPosition(tmpLevel, x + extraStones + 1, i, tile)
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

    def is_hazard(self, x, y):
        return self.inBounds(x, y) and self.tileAtPosition(x, y) == HAZARD

    def is_platform(self, x, y):
        return self.inBounds(x, y) and self.tileAtPosition(x, y) == PLATFORM

    def is_warp(self, x, y):
        return self.inBounds(x, y) and self.tileAtPosition(x, y) == WARP

    def is_spring(self, x, y):
        return self.inBounds(x, y) and self.tileAtPosition(x, y) == SPRING

    def at_warp(self, x=None, y=None):
        """True when a door or pipe is close enough to step into, from the agent's
        own cell or from (x, y). Adjacent counts: a warp is solid."""
        x = self.x if x is None else x
        y = self.y if y is None else y
        return any(self.is_warp(x + dx, y + dy)
                   for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)))

    def standable(self, x, y):
        """True when the cell below (x, y) holds the agent up."""
        return self.inBounds(x, y + 1) and (not self.passable(x, y + 1)
                                            or self.is_platform(x, y + 1)
                                            or self.is_spring(x, y + 1))

    def get_successor(self, action):
        """Return the next state, or None if the action is not legal here."""
        newJumpVelocity = self.jumpVelocity
        newMarioX = self.x
        newMarioY = self.y

        # Falling off bottom of screen 
        if not self.inBounds(self.x, self.y + 1):
            return None

        if newJumpVelocity == 0:  # Not mid-jump
            # A platform or spring below stops the fall, passable or not
            if (self.passable(newMarioX, newMarioY + 1)
                    and not self.is_platform(newMarioX, newMarioY + 1)
                    and not self.is_spring(newMarioX, newMarioY + 1)):  # Falling
                newMarioY += 1  # Fall down
            elif action.getDirection() == self.MarioAction.DIRECTION.JUMP:  # Start jump
                # Off a spring instead of off the floor
                newJumpVelocity = (SPRING_JUMP
                                   if self.is_spring(newMarioX, newMarioY + 1) else 5)
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

        # Coming to rest on a static hazard (spikes, ...) is death -- prune the state
        if newJumpVelocity == 0 and self.is_hazard(newMarioX, newMarioY + 1):
            return None

        return MM2State(self.level, newJumpVelocity, newMarioX, newMarioY)

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
        """True once Mario reaches the last x coordinate, or touches a warp, which
        leads out of the scene."""
        return self.isGoalX(self.x) or self.at_warp()

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
        if not isinstance(other, MM2State):
            return False
        return (self.jumpVelocity == other.jumpVelocity
                and self.x == other.x
                and self.y == other.y)

    def __str__(self):
        return f"({self.x},{self.y})"
