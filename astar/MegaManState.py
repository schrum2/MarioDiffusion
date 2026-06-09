from enum import Enum

MEGA_MAN_ASTAR_JUMP_HEIGHT = 4
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
FALL_STEPS_PER_SIDEWAYS_MOVE = 3
ONE_ENEMY_NULL = 9

class MegaManState:
    def __init__(self, level, x, y, orb, jump_velocity, fall_horizontal_mod_int):
        self.level = level; # [[int]] 2d array of tile types
        self.x = x
        self.y = y
        self.orb = orb
        self.jump_velocity = jump_velocity;
        self.fall_horizontal_mod_int = fall_horizontal_mod_int

    # distance to level orb
    def orb_heuristic(self):
        return max(abs(self.x - self.orb[1]), abs(self.y - self.orb[0]));

    
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
            return self.move == other.getMOVE()

        def to_string(self):
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
        assert self.inBounds(new_x, new_y)
        
        if not self.inBounds(self.x, self.y + 1):
            return None
        
        if ((self.inBounds(new_x, new_y - 1) or (new_y - 1 >= 0 and self.tileAtPosition(new_x, new_y - 1) == MEGA_MAN_TILE_HAZARD)) and self.inBounds(new_x, new_y + 1) and (not self.passable(new_x - 1, new_y + 1) or not self.passable(new_x + 1, new_y + 1)) and (not self.passable(new_x, new_y - 1) or self.tileAtPosition(new_x, new_y - 1) == MEGA_MAN_TILE_LADDER) and self.tileAtPosition(new_x, new_y) != MEGA_MAN_TILE_LADDER):
            sliding = True

        if self.tileAtPosition(new_x, new_y) == MEGA_MAN_TILE_LADDER:
            falling = False
            jumping = False
            new_fall_horizontal_mod_int = 0
            new_jump_velocity = 0
        
        if new_jump_velocity > 0:
            if (self.passable(new_x, new_y - 1) and self.tileAtPosition(new_x, new_y - 1) != MEGA_MAN_TILE_BREAKABLE or (self.inBounds(new_x, new_y) and self.tileAtPosition(new_x, new_y - 1) == MEGA_MAN_TILE_MOVING_PLATFORM)):
                jumping = True
                new_y -= 1
                new_jump_velocity -= 1
            else:
                new_jump_velocity = 0
                jumping = False
        
        if new_jump_velocity == 0:
            jumping = False

            if (((not sliding and self.passable(new_x, new_y + 1)) 
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
            if ((not jumping
                and (((falling or self.tileAtPosition(new_x, new_y) == MEGA_MAN_TILE_LADDER) and self.passable(new_x + 1, new_y) and self.passable(new_x + 1, new_y - 1) and new_fall_horizontal_mod_int & FALL_STEPS_PER_SIDEWAYS_MOVE == 0) or
                (self.tileAtPosition(new_x, new_y) != MEGA_MAN_TILE_LADDER and not falling and self.passable(new_x + 1, new_y) and (not self.passable(new_x, new_y + 1) or self.tileAtPosition(new_x, new_y + 1) == MEGA_MAN_TILE_LADDER or self.tileAtPosition(new_x, new_y + 1) == MEGA_MAN_TILE_MOVING_PLATFORM)))) or
                (jumping and self.passable(new_x + 1, new_y) and ((self.passable(new_x + 1, new_y - 1) and self.passable(new_x, new_y - 1)) or self.passable(new_x + 1, new_y + 1) and self.passable(new_x, new_y + 1)))):

                new_x += 1
            elif self.y == new_y:
                return None
    
        # left movement
        if action.getMOVE() == self.MegaManAction.MOVE.LEFT:
            if ((not jumping
                and (((falling or self.tileAtPosition(new_x, new_y) == MEGA_MAN_TILE_LADDER) and self.passable(new_x - 1, new_y) and self.passable(new_x - 1, new_y - 1) and new_fall_horizontal_mod_int & FALL_STEPS_PER_SIDEWAYS_MOVE == 0) or
                (self.tileAtPosition(new_x, new_y) != MEGA_MAN_TILE_LADDER and not falling and self.passable(new_x - 1, new_y) and (not self.passable(new_x, new_y + 1) or self.tileAtPosition(new_x, new_y + 1) == MEGA_MAN_TILE_LADDER or self.tileAtPosition(new_x, new_y + 1) == MEGA_MAN_TILE_MOVING_PLATFORM)))) or
                (jumping and self.passable(new_x - 1, new_y) and ((self.passable(new_x - 1, new_y - 1) and self.passable(new_x, new_y - 1)) or self.passable(new_x - 1, new_y + 1) and self.passable(new_x, new_y + 1)))):

                new_x -= 1
            elif self.y == new_y:
                return None
            
        # up movement (ladder)
        if action.getMOVE() == self.MegaManAction.MOVE.UP:
            if not sliding and self.inBounds(new_x, new_y - 1) and self.passable(new_x, new_y - 1) and self.tileAtPosition(new_x, new_y) == MEGA_MAN_TILE_LADDER and self.passable(new_x, new_y - 2):
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
    

    def noHazardBeneath(self, x, y):
        if self.tileAtPosition(x, y) != MEGA_MAN_TILE_HAZARD and self.tileAtPosition(x, y) <= 10:
            return True
        else: 
            return False
    

    def getSpawnFromVGLC(self):
        start = (-1, -1)
        tile = -1
        done = False
        i = 0
        while i < len(self.level) and not done:
            j = 0
            while j < len(self.level[i]) and not done:
                tile = self.level[i][j]
                if tile == MEGA_MAN_TILE_SPAWN:
                    start = (j, i)
                    self.level[i][j] = MEGA_MAN_TILE_EMPTY
                    done = True
        return start


    def getLegalActions(self, mmstate):
        valid_actions = []
        for move in self.MegaManAction.MOVE:
            if mmstate.getSuccessor(self.MegaManAction(move) != None):
                valid_actions.append(self.MegaManAction(move))
        return valid_actions
    

    def isGoal(self):
        return self.x == self.orb[1] and self.y == self.orb[0]


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
        return x >= 0 and x < len(self.level[0]) and y >= 0 and y < len(self.level) and self.level[y][x] != ONE_ENEMY_NULL  and self.noHazardBeneath(x, y)
    
    def tileAtPosition(self, x, y):
        return self.level[y][x]
    

    def passable(self, x, y):
        if not self.inBounds(x, y):
            return False
        tile = self.tileAtPosition(x, y)

        if (tile == MEGA_MAN_TILE_EMPTY or tile == MEGA_MAN_TILE_LADDER or tile == MEGA_MAN_TILE_ORB or tile == MEGA_MAN_TILE_BREAKABLE or tile == MEGA_MAN_TILE_WATER):
            return True
        
        return False
    
