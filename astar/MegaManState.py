from enum import Enum

MEGA_MAN_ASTAR_JUMP_HEIGHT = 5
MEGA_MAN_TILE_EMPTY = 0
MEGA_MAN_TILE_GROUND = 1
MEGA_MAN_TILE_LADDER = 2
MEGA_MAN_TILE_HAZARD = 3
MEGA_MAN_TILE_BREAKABLE = 4
MEGA_MAN_TILE_MOVING_PLATFORM = 5
MEGA_MAN_TILE_CANNON = 6
MEGA_MAN_TILE_ORB = 7
MEGA_MAN_TILE_NULL = 9
MEGA_MAN_TILE_SPAWN = 8
MEGA_MAN_TILE_WATER = 10
FOOTHOLDER_ENEMY = 27
FALL_STEPS_PER_SIDEWAYS_MOVE = 2
ONE_ENEMY_NULL = 9

class MegaManState:
    def __init__(self, level, x, y, orb, jump_velocity, fall_horizontal_mod_int):
        self.level = level; # [[int]] 2d array of tile types
        self.x = x
        self.y = y
        self.orb = orb
        self.jump_velocity = jump_velocity;
        self.fall_horizontal_mod_int = fall_horizontal_mod_int

    @classmethod
    def from_level(cls, level):
        """Build a start state from a level alone"""

        scanner = cls(level, 0, 0, (-1, -1), 0, 0)  # temporary, only to run the scanners
        spawn = scanner.getSpawnFromVGLC()          # (x, y); also blanks the spawn tile
        return cls.from_level_and_start(level, spawn)

    @classmethod
    def from_level_and_start(cls, level, start):
        """Build a start state from a level and an explicit (x, y) start point"""
        
        scanner = cls(level, 0, 0, (-1, -1), 0, 0)
        orb = scanner.find_orb()                    # (x, y)
        return cls(level, start[0], start[1], orb, 0, 0)

    # distance to level orb
    def orb_heuristic(self):
        return max(abs(self.x - self.orb[0]), abs(self.y - self.orb[1]))

    
    class MegaManAction:
        def __init__(self, move):
            self.move = move
        
        class MOVE(Enum):
            RIGHT = 0
            LEFT = 1
            UP = 2
            DOWN = 3
            JUMP = 4

        def getMOVE(self):
            return self.move

        def __eq__(self, other):
            return isinstance(other, MegaManState.MegaManAction) and self.move == other.move

        def __hash__(self):
            return hash(self.move)

        def __str__(self):
            if self.move == self.MOVE.RIGHT:
                return "RIGHT"
            elif self.move == self.MOVE.LEFT:
                return "LEFT"
            elif self.move == self.MOVE.UP:
                return "UP"
            elif self.move == self.MOVE.DOWN:
                return "DOWN"
            elif self.move == self.MOVE.JUMP:
                return "JUMP"
            
        

    # scan level to get orb position
    def find_orb(self):
        orb = (-1, -1)
        for y in range(len(self.level)):
            for x in range(len(self.level[y])):
                if self.level[y][x] == MEGA_MAN_TILE_ORB:
                    orb = (x, y)
        return orb
    
    def get_successor(self, action):
        new_jump_velocity = self.jump_velocity
        new_x = self.x
        new_y = self.y
        new_fall_horizontal_mod_int = self.fall_horizontal_mod_int
        falling = False
        jumping = False
        sliding = False
        skimming = False
        assert self.inBounds(new_x, new_y)

        # Falling off the bottom of the screen (into a gap) is deat EXCEPT when standing on a ladder
        if not self.inBounds(self.x, self.y + 1) and self.tileAtPosition(self.x, self.y) != MEGA_MAN_TILE_LADDER:
            return None

        if ((self.inBounds(new_x, new_y - 1) or (new_y - 1 >= 0 and self.tileAtPosition(new_x, new_y - 1) == MEGA_MAN_TILE_HAZARD)) and self.inBounds(new_x, new_y + 1) and (not self.passable(new_x - 1, new_y + 1) or not self.passable(new_x + 1, new_y + 1)) and (not self.passable(new_x, new_y - 1) or self.tileAtPosition(new_x, new_y - 1) == MEGA_MAN_TILE_LADDER) and self.tileAtPosition(new_x, new_y) != MEGA_MAN_TILE_LADDER):
            sliding = True

        if self.tileAtPosition(new_x, new_y) == MEGA_MAN_TILE_LADDER:
            falling = False
            jumping = False
            new_fall_horizontal_mod_int = 0
            new_jump_velocity = 0
        
        if new_jump_velocity > 0:
            if (self.passable(new_x, new_y - 1) and self.tileAtPosition(new_x, new_y - 1) != MEGA_MAN_TILE_BREAKABLE or (self.inBounds(new_x, new_y - 1) and self.tileAtPosition(new_x, new_y - 1) == MEGA_MAN_TILE_MOVING_PLATFORM)):
                jumping = True
                new_y -= 1
                new_jump_velocity -= 1
            else:
                new_jump_velocity = 0
                jumping = False
                skimming = True  # head bonk: permit a single apex skim instead of an instant drop

        # Apex skim (option 1): after a head bonk, allow one jump-speed sideways step at this
        # height rather than dropping a tile immediately and switching to the throttled fall.
        # Bounded to a single tile - the resulting state has zero velocity, so the next step
        # falls normally and no upward motion resumes.
        skim_right = (skimming and action.getMOVE() == self.MegaManAction.MOVE.RIGHT
                      and self.passable(new_x + 1, new_y) and self.passable(new_x + 1, new_y - 1))
        skim_left = (skimming and action.getMOVE() == self.MegaManAction.MOVE.LEFT
                     and self.passable(new_x - 1, new_y) and self.passable(new_x - 1, new_y - 1))
        
        if new_jump_velocity == 0:
            jumping = False

            if skim_right or skim_left:
                pass  # apex skim: hold height this step; the lateral block moves him sideways
            elif (((not sliding and self.passable(new_x, new_y + 1))
                or (sliding and self.passable(new_x, new_y + 1) and (self.passable(new_x - 1, new_y + 1) and self.tileAtPosition(new_x - 1, new_y + 1) != MEGA_MAN_TILE_LADDER or self.passable(new_x + 1, new_y + 1) and self.tileAtPosition(new_x + 1, new_y + 1) != MEGA_MAN_TILE_LADDER))) 
                and self.tileAtPosition(new_x, new_y + 1) != MEGA_MAN_TILE_LADDER and self.tileAtPosition(new_x, new_y + 1) != MEGA_MAN_TILE_BREAKABLE):
                
                new_y += 1
                new_fall_horizontal_mod_int += 1
                new_fall_horizontal_mod_int %= FALL_STEPS_PER_SIDEWAYS_MOVE
                falling = True
            
            elif not sliding and action.getMOVE() == self.MegaManAction.MOVE.JUMP and self.tileAtPosition(new_x, new_y) != MEGA_MAN_TILE_LADDER:
                new_jump_velocity = MEGA_MAN_ASTAR_JUMP_HEIGHT
        
        elif action.getMOVE() == self.MegaManAction.MOVE.JUMP:
            return None # can't jump mid-jump
        
        if not self.passable(new_x, new_y + 1) or (self.inBounds(new_x, new_y + 1) and self.tileAtPosition(new_x, new_y + 1) == MEGA_MAN_TILE_LADDER):
            falling = False
            new_fall_horizontal_mod_int = 0

        
        # right movement
        if action.getMOVE() == self.MegaManAction.MOVE.RIGHT:
            if (skim_right or (not jumping
                and (((falling or self.tileAtPosition(new_x, new_y) == MEGA_MAN_TILE_LADDER) and self.passable(new_x + 1, new_y) and self.passable(new_x + 1, new_y - 1) and new_fall_horizontal_mod_int % FALL_STEPS_PER_SIDEWAYS_MOVE == 0) or
                (self.tileAtPosition(new_x, new_y) != MEGA_MAN_TILE_LADDER and not falling and self.passable(new_x + 1, new_y) and (not self.passable(new_x, new_y + 1) or self.tileAtPosition(new_x, new_y + 1) == MEGA_MAN_TILE_LADDER or self.tileAtPosition(new_x, new_y + 1) == MEGA_MAN_TILE_MOVING_PLATFORM)))) or
                (jumping and self.passable(new_x + 1, new_y) and ((self.passable(new_x + 1, new_y - 1) and self.passable(new_x, new_y - 1)) or self.passable(new_x + 1, new_y + 1) and self.passable(new_x, new_y + 1)))):

                new_x += 1
            elif self.y == new_y:
                return None
    
        # left movement
        if action.getMOVE() == self.MegaManAction.MOVE.LEFT:
            if (skim_left or (not jumping
                and (((falling or self.tileAtPosition(new_x, new_y) == MEGA_MAN_TILE_LADDER) and self.passable(new_x - 1, new_y) and self.passable(new_x - 1, new_y - 1) and new_fall_horizontal_mod_int % FALL_STEPS_PER_SIDEWAYS_MOVE == 0) or
                (self.tileAtPosition(new_x, new_y) != MEGA_MAN_TILE_LADDER and not falling and self.passable(new_x - 1, new_y) and (not self.passable(new_x, new_y + 1) or self.tileAtPosition(new_x, new_y + 1) == MEGA_MAN_TILE_LADDER or self.tileAtPosition(new_x, new_y + 1) == MEGA_MAN_TILE_MOVING_PLATFORM)))) or
                (jumping and self.passable(new_x - 1, new_y) and ((self.passable(new_x - 1, new_y - 1) and self.passable(new_x, new_y - 1)) or self.passable(new_x - 1, new_y + 1) and self.passable(new_x, new_y + 1)))):

                new_x -= 1
            elif self.y == new_y:
                return None
            
        # up movement (ladder). Head clearance (passable two above) is normally required
        # because Mega Man is two tiles tall, but climbing up off the top of the screen is how you exit upward 
        if action.getMOVE() == self.MegaManAction.MOVE.UP:
            head_clear = self.passable(new_x, new_y - 2) or self.offScreen(new_x, new_y - 2)
            if not sliding and self.inBounds(new_x, new_y - 1) and self.passable(new_x, new_y - 1) and self.tileAtPosition(new_x, new_y) == MEGA_MAN_TILE_LADDER and head_clear:
                new_y -= 1
            else:
                return None
            
        # down movement (ladder)
        if action.getMOVE() == self.MegaManAction.MOVE.DOWN:
            if not sliding and self.inBounds(new_x, new_y + 1) and (self.tileAtPosition(new_x, new_y + 1) == MEGA_MAN_TILE_LADDER or self.tileAtPosition(new_x, new_y + 1) == MEGA_MAN_TILE_MOVING_PLATFORM):
                new_y += 1
            else:
                return None
            
        if not self.inBounds(new_x, new_y):
            return None
        

        result = MegaManState(self.level, new_x, new_y, self.orb, new_jump_velocity, new_fall_horizontal_mod_int)
        return result

    def get_successors(self):
        """List of (next_state, action, step_cost) reachable from this state.

        Mirrors State.getSuccessors in MM-NEAT; this is what the search loop consumes.
        """
        successors = []
        for a in self.getLegalActions(self):
            successor = self.get_successor(a)
            successors.append((successor, a, self.stepCost()))
        return successors

    def noHazardBeneath(self, x, y):
        if self.tileAtPosition(x, y) != MEGA_MAN_TILE_HAZARD and self.tileAtPosition(x, y) <= 10:
            return True
        else: 
            return False
    

    def addOrb(self):
        """
        Scans the level for a suitable orb placement for the heuristic to target when no orb is present

        Necessary for level snippets/scenes that don't naturally contain an orb
        """
        for x in range(len(self.level[0])-1, -1, -1): # start from the right
            for y in range(len(self.level)): # start from the top
                curr_tile = self.level[y][x]
                feet_tile = self.level[y-1][x]
                head_tile = self.level[y-2][x] if y - 2 >= 0 else None
                if (y - 2 >= 0
                                and (curr_tile == MEGA_MAN_TILE_LADDER or
                                    curr_tile == MEGA_MAN_TILE_GROUND or
                                    curr_tile == MEGA_MAN_TILE_MOVING_PLATFORM)
                                and (feet_tile == MEGA_MAN_TILE_EMPTY or
                                     feet_tile == MEGA_MAN_TILE_WATER)
                                and (head_tile == MEGA_MAN_TILE_EMPTY or
                                     head_tile == MEGA_MAN_TILE_WATER)):
                    self.level[y-1][x] = MEGA_MAN_TILE_ORB
                    return True

        return False # suitable orb location not found
    

    def forceOrb(self):
        """Fallback when addOrb finds no natural ledge: scan right to left, top to bottom
        to the first non-null cell and insert orb there."""

        for x in range(len(self.level[0]) - 1, -1, -1): # right to left
            for y in range(len(self.level) - 1, 1, -1):  # bottom to top (y-2 >= 0)
                if self.level[y][x] != MEGA_MAN_TILE_NULL:
                    self.level[y][x]     = MEGA_MAN_TILE_GROUND
                    self.level[y - 1][x] = MEGA_MAN_TILE_ORB
                    self.level[y - 2][x] = MEGA_MAN_TILE_EMPTY
                    return True
        return False  # no non-null cell with headroom found

    def placeSpawn(self):
        """Same standable-ledge logic as addOrb (incl. the two-tile-tall head clearance
        requirement, non-destructive), but scans left to right/bottom to top"""

        for x in range(len(self.level[0])):              # left to right
            for y in range(len(self.level) - 1, -1, -1): # bottom to top
                curr_tile = self.level[y][x]
                feet_tile = self.level[y - 1][x]
                head_tile = self.level[y - 2][x] if y - 2 >= 0 else None
                if (y - 2 >= 0
                              and (curr_tile == MEGA_MAN_TILE_LADDER or
                                   curr_tile == MEGA_MAN_TILE_GROUND or
                                   curr_tile == MEGA_MAN_TILE_MOVING_PLATFORM)
                              and (feet_tile == MEGA_MAN_TILE_EMPTY or
                                   feet_tile == MEGA_MAN_TILE_WATER)
                              and (head_tile == MEGA_MAN_TILE_EMPTY or
                                   head_tile == MEGA_MAN_TILE_WATER)):
                    self.level[y - 1][x] = MEGA_MAN_TILE_SPAWN
                    return True
        return False  # suitable spawn location not found

    def forceSpawn(self):
        """same idea as forceOrb, but scans left to right """
        
        for x in range(len(self.level[0])): # left to right
            for y in range(len(self.level) - 1, 1, -1):  # bottom to top (y-2 >= 0)
                if self.level[y][x] != MEGA_MAN_TILE_NULL:
                    self.level[y][x]     = MEGA_MAN_TILE_GROUND
                    self.level[y - 1][x] = MEGA_MAN_TILE_SPAWN
                    self.level[y - 2][x] = MEGA_MAN_TILE_EMPTY
                    return True
        return False  # no non-null cell with headroom found

    def getSpawnFromVGLC(self):
        """Find the spawn tile, replace it with empty space, and return it as (x, y).

        NOTE: mutates self.level (matches the Java version).
        """
        start = (-1, -1)
        for i in range(len(self.level)):
            for j in range(len(self.level[i])):
                if self.level[i][j] == MEGA_MAN_TILE_SPAWN:
                    start = (j, i)
                    self.level[i][j] = MEGA_MAN_TILE_EMPTY
                    return start
        return start


    def getLegalActions(self, mmstate):
        valid_actions = []
        for move in self.MegaManAction.MOVE:
            if mmstate.get_successor(self.MegaManAction(move)) is not None:
                valid_actions.append(self.MegaManAction(move))
        return valid_actions
    

    def isGoal(self):
        return self.x == self.orb[0] and self.y == self.orb[1]


    def  __hash__(self):
        prime = 31
        result = 1
        result = prime * result + self.fall_horizontal_mod_int
        result = prime * result + self.x
        result = prime * result + self.y
        result = prime * result + self.jump_velocity
        return result
	
    
    def stepCost(self):
        return 1
    
    def __eq__(self, other):
        if self is other:
            return True
        if not other:
            return False
        if not isinstance(other, MegaManState):
            return False
        if other.x != self.x or other.y != self.y or other.jump_velocity != self.jump_velocity or other.fall_horizontal_mod_int != self.fall_horizontal_mod_int:
            return False
        if (not self.orb and other.orb) or self.orb != other.orb : # this dude has no orb but the other dude does, or if the two orbs are different
            return False
        return True
    

    def __str__(self):
        return f"({self.x}, {self.y})"  

    
    def inBounds(self, x, y):
        return x >= 0 and y >= 0 and y < len(self.level) and x < len(self.level[y]) and self.level[y][x] != ONE_ENEMY_NULL  and self.noHazardBeneath(x, y)

    def offScreen(self, x, y):
        """True if (x, y) is outside the playable area: off the grid, or NULL padding."""
        return (y < 0 or x < 0 or y >= len(self.level) or x >= len(self.level[y])
                or self.level[y][x] == ONE_ENEMY_NULL)

    def tileAtPosition(self, x, y):
        return self.level[y][x]
    

    def passable(self, x, y):
        if not self.inBounds(x, y):
            return False
        tile = self.tileAtPosition(x, y)

        if (tile == MEGA_MAN_TILE_EMPTY or tile == MEGA_MAN_TILE_LADDER or tile == MEGA_MAN_TILE_ORB or tile == MEGA_MAN_TILE_BREAKABLE or tile == MEGA_MAN_TILE_WATER):
            return True
        
        return False
    
