from collections import deque
from enum import Enum

LODE_RUNNER_TILE_EMPTY = 0
LODE_RUNNER_TILE_GOLD = 1
LODE_RUNNER_TILE_ENEMY = 2
LODE_RUNNER_TILE_DIGGABLE = 3
LODE_RUNNER_TILE_LADDER = 4
LODE_RUNNER_TILE_ROPE = 5
LODE_RUNNER_TILE_GROUND = 6
LODE_RUNNER_TILE_SPAWN = 7

# Big cost to discourage moving sideways through diggable ground
SIDEWAYS_DIG_COST_MULTIPLIER = 100

# MST weights memoized per gold frozenset: the same gold subset recurs at many player
# positions during a search, so this is hit constantly. Capped so a long batch of
# levels can't grow it without bound.
_MST_CACHE = {}
_MST_CACHE_LIMIT = 200000


def _mst_weight(points):
    """Weight of a minimum spanning tree over points under the Manhattan metric (Prim's)."""
    cached = _MST_CACHE.get(points)
    if cached is not None:
        return cached
    pts = list(points)
    n = len(pts)
    total = 0
    if n > 1:
        best = [float('inf')] * n   # cheapest connection of each point to the tree
        best[0] = 0
        in_tree = [False] * n
        for _ in range(n):
            u = min((i for i in range(n) if not in_tree[i]), key=best.__getitem__)
            in_tree[u] = True
            total += best[u]
            ux, uy = pts[u]
            for v in range(n):
                if not in_tree[v]:
                    d = abs(ux - pts[v][0]) + abs(uy - pts[v][1])
                    if d < best[v]:
                        best[v] = d
    if len(_MST_CACHE) >= _MST_CACHE_LIMIT:
        _MST_CACHE.clear()
    _MST_CACHE[points] = total
    return total


class LodeRunnerState:

    def __init__(self, level, goldLeft, currentX, currentY, allowWeirdMoves=False):
        self.level = level                # [[int]] 2d array of tile types
        self.goldLeft = goldLeft          # frozenset of (x, y) tuples, makes it hashable
        self.currentX = currentX
        self.currentY = currentY
        self.allowWeirdMoves = allowWeirdMoves

    @classmethod
    def from_level(cls, level, allowWeirdMoves=False):
        """Build a start state from a level alone"""

        start = cls.getSpawnFromVGLC(level)   # (x, y); blanks the spawn tile
        gold = cls.fillGold(level)            # frozenset; blanks the gold tiles
        return cls(level, gold, start[0], start[1], allowWeirdMoves)

    @classmethod
    def from_level_and_start(cls, level, start, allowWeirdMoves=False):
        """Build a start state from a level and an explicit (x, y) start point"""

        gold = cls.fillGold(level)
        return cls(level, gold, start[0], start[1], allowWeirdMoves)

    # Heuristic: Manhattan distance from the player to the farthest remaining gold.
    # (MM-NEAT's heuristic; kept for reference, but mstToRemainingGold dominates it.)
    def manhattanToFarthestGold(self):
        maxDistance = 0
        for (px, py) in self.goldLeft:
            distance = abs(self.currentX - px) + abs(self.currentY - py)
            maxDistance = max(maxDistance, distance)
        return maxDistance

    # Heuristic: minimum-spanning-tree weight of the remaining gold (Manhattan metric)
    # plus the distance to the nearest remaining gold. Any path that collects every
    # gold must walk from the player to some first gold (>= nearest distance) and then
    # visit all the rest (its consecutive-visit distances form a spanning path of the
    # gold, >= the MST weight). Every move covers at most 1 tile of Manhattan distance
    # at cost >= 1, so this is admissible -- and far stronger than the farthest-gold
    # bound, which is what made levels with many golds blow the search budget: states
    # are (position, gold subset) pairs, and a weak heuristic lets A* churn through
    # thousands of collection orders that an MST bound prunes immediately.
    def mstToRemainingGold(self):
        if not self.goldLeft:
            return 0
        nearest = min(abs(self.currentX - px) + abs(self.currentY - py)
                      for (px, py) in self.goldLeft)
        return nearest + _mst_weight(self.goldLeft)

    class LodeRunnerAction:

        def __init__(self, move):
            self.movement = move

        class MOVE(Enum):
            RIGHT = 0
            LEFT = 1
            UP = 2
            DOWN = 3

        def getMove(self):
            return self.movement

        def __eq__(self, other):
            if isinstance(other, LodeRunnerState.LodeRunnerAction):
                return other.movement == self.movement
            return False

        def __hash__(self):
            return hash(self.movement)

        def __str__(self):
            return str(self.movement)

    @staticmethod
    def fillGold(level):
        """Record where the gold is, then blank those tiles to empty"""

        gold = set()
        for i in range(len(level)):
            for j in range(len(level[i])):
                if level[i][j] == LODE_RUNNER_TILE_GOLD:
                    gold.add((j, i))
                    level[i][j] = LODE_RUNNER_TILE_EMPTY
        return frozenset(gold)

    @staticmethod
    def getSpawnFromVGLC(level):
        """Find the spawn tile, blank it to empty, and return it as (x, y)"""

        start = (0, 0)
        for i in range(len(level)):
            for j in range(len(level[i])):
                if level[i][j] == LODE_RUNNER_TILE_SPAWN:
                    start = (j, i)
                    level[i][j] = LODE_RUNNER_TILE_EMPTY
                    return start
        return start

    def get_successor(self, action):
        """Return the next state, or None if the action is not legal here"""

        newX = self.currentX
        newY = self.currentY
        assert self.inBounds(newX, newY)
        move = action.getMove()
        MOVE = self.LodeRunnerAction.MOVE

        if move == MOVE.RIGHT:
            beneath = -1 if not self.inBounds(newX, newY + 1) else self.tileAtPosition(newX, newY + 1)
            if self.passable(newX + 1, newY) and \
                    (self.tileAtPosition(newX, newY) == LODE_RUNNER_TILE_ROPE or
                     self.tileAtPosition(newX, newY) == LODE_RUNNER_TILE_LADDER):
                newX += 1
            elif (self.allowWeirdMoves and
                  self.inBounds(newX + 1, newY) and
                  (beneath == -1 or beneath == LODE_RUNNER_TILE_LADDER or
                   beneath == LODE_RUNNER_TILE_DIGGABLE or beneath == LODE_RUNNER_TILE_GROUND) and
                  self.tileAtPosition(newX + 1, newY) == LODE_RUNNER_TILE_DIGGABLE and
                  self.diggablePath(newX + 1, newY)):
                # Weird case: move sideways through diggable ground (the player could
                # hypothetically have dug the ground above to make this possible).
                newX += 1
            elif (self.tileAtPosition(newX, newY) != LODE_RUNNER_TILE_LADDER and
                  beneath != -1 and
                  beneath != LODE_RUNNER_TILE_LADDER and
                  beneath != LODE_RUNNER_TILE_DIGGABLE and
                  beneath != LODE_RUNNER_TILE_GROUND):  # no ground under the player
                return None  # cannot move right
            elif self.passable(newX + 1, newY):
                newX += 1
            else:
                return None

        elif move == MOVE.LEFT:
            # Turns out you can walk on the bottom of the screen with nothing beneath you
            beneath = -1 if not self.inBounds(newX, newY + 1) else self.tileAtPosition(newX, newY + 1)
            if self.passable(newX - 1, newY) and \
                    (self.tileAtPosition(newX, newY) == LODE_RUNNER_TILE_ROPE or
                     self.tileAtPosition(newX, newY) == LODE_RUNNER_TILE_LADDER):
                newX -= 1
            elif (self.allowWeirdMoves and
                  self.inBounds(newX - 1, newY) and
                  (beneath == -1 or beneath == LODE_RUNNER_TILE_LADDER or
                   beneath == LODE_RUNNER_TILE_DIGGABLE or beneath == LODE_RUNNER_TILE_GROUND) and
                  self.tileAtPosition(newX - 1, newY) == LODE_RUNNER_TILE_DIGGABLE and
                  self.diggablePath(newX - 1, newY)):
                newX -= 1
            elif (self.tileAtPosition(newX, newY) != LODE_RUNNER_TILE_LADDER and
                  beneath != -1 and
                  beneath != LODE_RUNNER_TILE_LADDER and
                  beneath != LODE_RUNNER_TILE_DIGGABLE and
                  beneath != LODE_RUNNER_TILE_GROUND):  # no ground under the player
                return None  # fall down
            elif self.passable(newX - 1, newY):
                newX -= 1
            else:
                return None

        elif move == MOVE.UP:
            if ((self.passable(newX, newY - 1) or  # Do not allow moving up ladders into solid tiles
                 (self.allowWeirdMoves and self.inBounds(newX, newY - 1) and
                  self.tileAtPosition(newX, newY - 1) == LODE_RUNNER_TILE_DIGGABLE)) and  # except diggable if weird moves allowed
                    self.inBounds(newX, newY - 1) and
                    self.tileAtPosition(newX, newY) == LODE_RUNNER_TILE_LADDER):  # be on a ladder to climb
                newY -= 1
            else:
                return None

        elif move == MOVE.DOWN:
            # Might descend if the resulting location is in bounds and not solid ground
            if self.inBounds(newX, newY + 1) and \
                    self.tileAtPosition(newX, newY + 1) != LODE_RUNNER_TILE_GROUND:
                # Special case for diggable ground: verify it was possible to dig out the square
                if (self.tileAtPosition(newX, newY + 1) != LODE_RUNNER_TILE_DIGGABLE or
                        # Diggable, but tile to left was not empty, providing a platform for digging
                        (self.inBounds(newX - 1, newY + 1) and
                         self.tileAtPosition(newX - 1, newY + 1) != LODE_RUNNER_TILE_EMPTY) or
                        # Diggable, but tile to right was not empty, providing a platform for digging
                        (self.inBounds(newX + 1, newY + 1) and
                         self.tileAtPosition(newX + 1, newY + 1) != LODE_RUNNER_TILE_EMPTY)):
                    newY += 1
                else:
                    return None
            else:
                # Standing on solid ground (or the bottom of the level): DOWN is not a
                # legal move. Without this the action becomes a "no-op" that returns a
                # state identical to its parent, flooding the frontier with duplicates.
                return None


        # Collect any gold at the new position by removing it from the set
        newGoldLeft = frozenset(p for p in self.goldLeft if p != (newX, newY))

        assert self.inBounds(newX, newY)
        return LodeRunnerState(self.level, newGoldLeft, newX, newY, self.allowWeirdMoves)

    def diggablePath(self, x, y):
        """True if this spot might be reachable by digging straight down from above"""
        while self.inBounds(x, y) and self.tileAtPosition(x, y) == LODE_RUNNER_TILE_DIGGABLE:
            y -= 1  # Move up
        return self.passable(x, y)

    def get_successors(self):
        """List of (next_state, action, step_cost) reachable from current state"""
        # Build successors directly rather than via getLegalActions, which would
        # recompute every successor a second time.
        successors = []
        for move in self.LodeRunnerAction.MOVE:
            a = self.LodeRunnerAction(move)
            successor = self.get_successor(a)
            if successor is not None:
                successors.append((successor, a, self.stepCost(a)))
        return successors

    def reachable_tree(self):
        """Flood-fill (BFS) from the player over POSITION space, ignoring gold.

        A gold cell is reachable iff it appears in this tree, so checking every gold
        for membership replaces the old collect-all A* search (which explored the
        (position x remaining-gold subset) space) with a single linear flood fill.
        """
        root = (self.currentX, self.currentY)
        parent = {root: None}
        queue = deque([self])
        while queue:
            state = queue.popleft()
            for succ, _action, _cost in state.get_successors():
                pos = (succ.currentX, succ.currentY)
                if pos not in parent:
                    parent[pos] = (state.currentX, state.currentY)
                    queue.append(succ)
        return parent

    def getLegalActions(self, state):
        """A list of valid actions for playing Lode Runner levels"""
        validActions = []
        for move in self.LodeRunnerAction.MOVE:
            if state.get_successor(self.LodeRunnerAction(move)) is not None:
                validActions.append(self.LodeRunnerAction(move))
        return validActions

    def passable(self, x, y):
        """True if the tile can be moved into (empty, enemy, ladder, or rope)"""
        if not self.inBounds(x, y):
            return False  # fail for bad bounds before tileAtPosition check
        tile = self.tileAtPosition(x, y)
        return (tile == LODE_RUNNER_TILE_EMPTY or tile == LODE_RUNNER_TILE_ENEMY or
                tile == LODE_RUNNER_TILE_LADDER or tile == LODE_RUNNER_TILE_ROPE)

    def inBounds(self, x, y):
        return y >= 0 and x >= 0 and y < len(self.level) and x < len(self.level[0])

    def tileAtPosition(self, x, y):
        return self.level[y][x]

    def isGoal(self):
        """True once there is no gold left"""
        return len(self.goldLeft) == 0

    # Dominance pruning hooks (see AStarSearch.search). Movement legality depends only
    # on the level and the player's position -- never on the gold collected -- so a
    # state at the same position holding a subset of the remaining gold can do
    # everything this one can. Pruning the dominated state is what tames the
    # (position x remaining-gold subsets) explosion in levels with many golds.
    def dominance_key(self):
        return (self.currentX, self.currentY)

    def dominates(self, other):
        """True if this state is at least as good as other (same position assumed):
        everything other still has to collect, this state must collect too."""
        return self.goldLeft <= other.goldLeft

    def stepCost(self, action):
        """Cost of taking the given action from this state (every move is one tile, except
        sideways/downward moves through diggable ground, which are penalized)"""
        move = action.getMove()
        MOVE = self.LodeRunnerAction.MOVE
        beneath = -1 if not self.inBounds(self.currentX, self.currentY + 1) else self.tileAtPosition(self.currentX, self.currentY + 1)

        # Moving sideways through diggable ground assumes the ground above was dug out first
        if (move == MOVE.LEFT and
                self.inBounds(self.currentX - 1, self.currentY) and
                self.tileAtPosition(self.currentX - 1, self.currentY) == LODE_RUNNER_TILE_DIGGABLE):
            cost = 1
            y = self.currentY
            x = self.currentX - 1
            while self.inBounds(x, y) and self.tileAtPosition(x, y) == LODE_RUNNER_TILE_DIGGABLE:
                cost += 1
                y -= 1  # Move up; getSuccessor guarantees we eventually reach an empty tile
            if not self.inBounds(x, y):
                cost = float('inf')  # Impossible action
            return cost * cost * SIDEWAYS_DIG_COST_MULTIPLIER
        elif (move == MOVE.RIGHT and
                self.inBounds(self.currentX + 1, self.currentY) and
                self.tileAtPosition(self.currentX + 1, self.currentY) == LODE_RUNNER_TILE_DIGGABLE):
            cost = 1
            y = self.currentY
            x = self.currentX + 1
            while self.inBounds(x, y) and self.tileAtPosition(x, y) == LODE_RUNNER_TILE_DIGGABLE:
                cost += 1
                y -= 1
            if not self.inBounds(x, y):
                cost = float('inf')
            return cost * cost * SIDEWAYS_DIG_COST_MULTIPLIER
        elif move == MOVE.DOWN and beneath == LODE_RUNNER_TILE_DIGGABLE:
            # Digging down is expensive: move aside, dig, move back, then fall.
            return 4
        else:
            return 1

    def __hash__(self):
        prime = 31
        result = 1
        result = prime * result + self.currentX
        result = prime * result + self.currentY
        result = prime * result + (0 if self.goldLeft is None else hash(self.goldLeft))
        return result

    def __eq__(self, other):
        if self is other:
            return True
        if other is None:
            return False
        if not isinstance(other, LodeRunnerState):
            return False
        return (self.currentX == other.currentX and
                self.currentY == other.currentY and
                self.goldLeft == other.goldLeft)

    def __str__(self):
        return f"Size:{len(self.goldLeft)} ({self.currentX}, {self.currentY})"
