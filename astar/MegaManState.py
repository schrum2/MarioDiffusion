from enum import Enum

MEGA_MAN_ASTAR_JUMP_HEIGHT = 4
MEGA_MAN_TILE_EMPTY = 0;
MEGA_MAN_TILE_GROUND = 1;
MEGA_MAN_TILE_LADDER = 2;
MEGA_MAN_TILE_HAZARD = 3;
MEGA_MAN_TILE_BREAKABLE = 4;
MEGA_MAN_TILE_MOVING_PLATFORM = 5;
MEGA_MAN_TILE_CANNON = 6;
MEGA_MAN_TILE_ORB = 7;
MEGA_MAN_TILE_NULL = 9;
MEGA_MAN_TILE_SPAWN = 8;
MEGA_MAN_TILE_WATER = 10;
FOOTHOLDER_ENEMY = 27;
FALL_STEPS_PER_SIDEWAYS_MOVE = 3;

class MegaManState:
    def __init__(self, level, x, y, orb, jump_velocity, fall_horizontal_mod_int):
        self.level = level; # [[int]] 2d array of tile types
        self.x = x
        self.y = y
        self.orbx = orb
        self.jump_velocity = jump_velocity;
        self.fall_horizontal_mod_int = fall_horizontal_mod_int

    # distance to level orb
    def orb_heuristic(self):
        return max(abs(self.x - self.orbx), abs(self.y - self.orby));

    
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

        def equals(self, other):
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
                    orb = (y, x)
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
                new_jump_velocity = Parameters.parameters.integerParameter("megaManAStarJumpHeight")
        
        elif action.getMOVE() == self.MegaManAction.MOVE.JUMP:
            return None # can't jump mid-jump
        
    
            
           
        

    
    def inBounds(self, x, y):
        return x >= 0 and x < len(self.level[0]) and y >= 0 and y < len(self.level)
    
    def tileAtPosition(self, x, y):
        return self.level[y][x]
    

    def passable(self, x, y):
        if not self.inBounds(x, y):
            return False
        tile = self.tileAtPosition(x, y)

        if (tile == MEGA_MAN_TILE_EMPTY or tile == MEGA_MAN_TILE_LADDER or tile == MEGA_MAN_TILE_ORB or tile == MEGA_MAN_TILE_BREAKABLE or tile == MEGA_MAN_TILE_WATER):
            return True
        
        return False