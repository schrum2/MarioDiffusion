"""Convert an MM2 level JSON (toost's export) to a Super Mario Maker: World
Engine (.swe) save.

A .swe is base64(UTF-8 JSON) + 40 hex chars of HMAC-SHA1 over the base64 text
(key "2559F35097-2021", from EngineTribe's SMMWESaveDecryptor) + a NUL.
The JSON is {"S0": overworld, "SB1": subworld}, each world split into:

    S1 metadata   S2 ground   S3 decorations (unused)   S4 objects
    S5 pipes      S6 cannons  S7 stretchy sprites       S8 door pairs

MM2 ground is in tiles, y up from the bottom; MM2 objects are in 160-subpixel
units, centered. SWE uses pixels (16/tile), y down from the top of a 27-tile
playfield: yy = (27 - 1 - row) * 16.

Best-effort conversion: objects SMMWE lacks are dropped with a warning, most
flag bits (wings, parachutes, ...) don't carry over, ground autotiles are
recomputed from neighbour occupancy, slopes become solid ground. Constants
marked "verified" come from saves placed by hand in the SMMWE editor.

Usage:
    python -m mm2pipeline_data.swe --input bcd_levels/json/<id>_overworld.json
    python -m mm2pipeline_data.swe --input <file or folder> -o <out> --user <smmwe name>

A matching *_subworld.json next to the overworld file becomes SB1.
"""

import argparse
import base64
import hashlib
import hmac
import json
from datetime import datetime
from pathlib import Path

from .ascii import repair_semisolid_cells

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SWE_HMAC_KEY      = b"2559F35097-2021"   # EngineTribe SMMWESaveDecryptor key
FIELD_HEIGHT_TILES = 27                  # playfield height (432 px / 16)
PX                = 16                    # pixels per tile in SWE
SUBPX             = 160                   # sub-pixels per tile in MM2 object coords

# gamestyle_raw (MM2) -> SWE gamestyle int. SMMWE has SMB1/SMB3/SMW/NSMBU;
# SM3DW has no SMMWE equivalent, so it falls back to NSMBU.
GAMESTYLE_MAP = {
    12621: 0,   # SMB1
    13133: 1,   # SMB3
    22349: 2,   # SMW
    21847: 3,   # NSMBU
    22323: 3,   # SM3DW -> NSMBU (closest)
}

# theme_raw (MM2, 0-9) -> SWE gametheme string.
THEME_MAP = {
    0: "overworld",
    1: "underground",
    2: "castle",
    3: "airship",
    4: "underwater",
    5: "ghost",
    6: "snow",
    7: "desert",
    8: "sky",
    9: "forest",
}

# 4-neighbour occupancy mask (N=1, E=2, S=4, W=8) -> ground autotile slot,
# from a sample level. The slot is theme-independent.
GROUND_AUTOTILE = {
    0:  2,    # isolated block
    1:  21,   # N only          (bottom cap of a vertical strip)
    2:  9,    # E only          (left cap of a horizontal strip)
    3:  18,   # N+E             (bottom-left outer corner)
    4:  53,   # S only          (top cap / grass top)
    5:  45,   # N+S             (vertical middle)
    6:  9,    # E+S             (top-left corner)
    7:  61,   # N+E+S           (left edge)
    8:  3,    # W only          (right cap of a horizontal strip)
    9:  11,   # N+W             (bottom-right outer corner)
    10: 54,   # E+W             (horizontal middle)
    11: 43,   # N+E+W           (bottom edge)
    12: 10,   # S+W             (top-right corner)
    13: 62,   # N+S+W           (right edge)
    14: 53,   # E+S+W           (top edge / grass top)
    15: 43,   # all four        (interior fill)
}

# MM2 object id -> SWE obj_*_res. None = no SMMWE equivalent, dropped with a
# warning. "approx" entries pick the closest available object.
OBJ_ID_MAP = {
    0:   "obj_goomba_res",          # Goomba
    1:   "obj_koopa_res",           # Koopa Troopa
    2:   "obj_pplant_res",          # Piranha Plant
    3:   "obj_hammerbro_res",       # Hammer Bro
    4:   "obj_block_res",           # Brick Block
    5:   "obj_qblock_res",          # Question Block
    6:   "obj_rock_res",            # Hard Block (verified: MM2 "Stone" id=6)
    7:   "obj_rock_res",            # Ground-as-object (approx; terrain is S2)
    8:   "obj_coin_res",            # Coin
    9:   None,                      # Pipe -> emitted to S5 (see build_pipes)
    10:  "obj_spring_res",          # Spring / Trampoline
    # 11 Lift -> S7 "obj_platform_res", see build_platform_objects
    12:  "obj_thwomp_res",          # Thwomp
    # 13 Bullet Bill Blaster -> S7 "obj_bullebill_base_res", see build_platform_objects
    # 14 Mushroom Platform -> S7 "obj_mushroom_platform_res", see build_platform_objects
    14:  None,
    15:  "obj_bobomb_res",          # Bob-omb
    # 16 Semisolid Platform -> S7 "obj_semisolid_platform1", see build_platform_objects
    # 17 Bridge -> S7 "obj_puente_res", see build_platform_objects / PLATFORM_S7_IDS
    17:  None,
    18:  "obj_pswitch_res",         # P Switch
    19:  "obj_pow_res",             # POW Block
    20:  "obj_mushroom_res",        # Super Mushroom
    21:  "obj_donut_res",           # Donut Block
    22:  "obj_nube_res",            # Cloud
    23:  "obj_noteblock_res",       # Note Block
    24:  "obj_firebar_res",         # Fire Bar
    25:  "obj_spiny_res",           # Spiny
    26:  None,                      # Goal Ground (terrain; goal is in S1)
    27:  None,                      # Goal (flagpole; stored in S1 goal_x/y)
    28:  "obj_buzzybeetle_res",     # Buzzy Beetle
    29:  "obj_block_hidden_res",    # Hidden Block
    30:  "obj_lakitu_res",          # Lakitu
    31:  "obj_clown_res",           # Lakitu's Cloud -> Clown Car (nearest rideable vehicle)
    32:  "obj_billbanzai_res",      # Banzai Bill
    33:  "obj_1up_res",             # 1-Up Mushroom
    34:  "obj_fireflower_res",      # Fire Flower
    35:  "obj_star_res",            # Super Star
    36:  "obj_lava_lift_res",       # Lava Lift
    37:  None,                      # Starting Brick (editor marker)
    38:  "obj_arrow_res",           # Starting Arrow
    39:  "obj_magikoopa_res",       # Magikoopa
    40:  "obj_spiny_res",           # Spike Top -> Spiny (approx)
    41:  "obj_boo_res",             # Boo
    42:  None,                      # Clown Car -> obj_clown_res / obj_clown_fire_res, see ONOFF_SWE_IDS
    43:  "obj_pinchos_res",         # Spike Trap
    44:  None,                      # Style Power-up A -> see STYLE_POWERUP_A_MAP (gamestyle-dependent)
    45:  "obj_egg_res",              # Shoe Goomba / Yoshi's Egg -> obj_egg_res (green Yoshi); works for SMB1/SMB3/SMW/NSMBU
    46:  "obj_drybones_res",        # Dry Bones
    # 47 Cannon -> S6 "obj_cannon_res", see build_cannons
    48:  "obj_blooper_res",         # Blooper
    # 49 Castle Bridge (approx) -> S7 "obj_puente_res", see build_platform_objects / PLATFORM_S7_IDS
    49:  None,
    50:  "obj_spring_res",          # Jumping Machine (approx)
    51:  None,                      # Skipsqueak
    52:  None,                      # Wiggler
    53:  "obj_cinta_res",           # Fast Conveyor Belt
    54:  "obj_soplete_res",         # Burner
    55:  None,                      # Door -> S8 paired entry, see build_doors
    56:  "obj_cheepcheep_res",      # Cheep Cheep
    57:  "obj_muncher_res",         # Muncher
    58:  "obj_rocky_res",           # Rocky Wrench
    59:  "obj_rails_res",           # Track / rail
    60:  "obj_podoboo_res",         # Lava Bubble
    61:  "obj_chomp_res",           # Chain Chomp
    62:  "obj_bowser_res",          # Bowser
    63:  "obj_ice_res",             # Ice Block
    64:  "obj_vine_res",            # Vine
    65:  None,                      # Stingby
    66:  "obj_arrow_res",           # Arrow (decoration)
    67:  "obj_oneway_res",          # One-Way Wall
    68:  "obj_grinder_res",         # Saw
    69:  None,                      # Player spawn (stored in S1 start_y)
    70:  "obj_coin10_res",          # Big Coin (10-coin)
    71:  "obj_platform_res",        # Half-Collision Platform (approx)
    72:  "obj_clown_res",           # Koopa Car (approx)
    73:  None,                      # Cinobio
    74:  "obj_spike_ball_res",      # Spike Ball
    75:  "obj_rock_res",            # Stone Block
    76:  "obj_torbellino_res",      # Twister
    77:  "obj_boomboom_res",        # Boom Boom
    78:  "obj_pokey_res",           # Pokey
    79:  "obj_pblock_res",          # P Block
    80:  "obj_expandplatf_res",     # Sprint Platform (approx)
    # 81 Style Power-up B -> see STYLE_POWERUP_B_MAP (gamestyle-dependent)
    82:  "obj_donut_res",           # Donut Block Platform
    83:  "obj_skewer_res",          # Skewer
    84:  None,                      # Snake Block
    85:  None,                      # Track Block
    86:  "obj_floruga_res",         # Charvaargh (approx)
    87:  None,                      # Slight Slope (terrain)
    88:  None,                      # Steep Slope (terrain)
    89:  None,                      # Reel Camera (cutscene marker)
    90:  "obj_checkpoint_res",      # Checkpoint Flag
    # 91 Seesaw -> S7 stretchy "obj_seesaw_res", sized to the seesaw's
    # footprint, see PLATFORM_S7_IDS / build_platform_objects
    91:  None,
    92:  "obj_pink_coin_res",       # Red Coin (approx -> pink coin)
    93:  None,                      # Clear Pipe
    94:  "obj_cinta_res",           # Conveyor Belt
    95:  "obj_key_res",             # Key
    96:  None,                      # Ant Trooper
    97:  None,                      # Warp Box
    98:  "obj_bowserjr_res",        # Bowser Jr.
    99:  None,                      # ON/OFF Block -> obj_onoffblock_res/_blue_res, see ONOFF_SWE_IDS
    100: None,                      # Dotted-Line Block -> obj_onoffplatform_res/_blue_res, see ONOFF_SWE_IDS
    101: None,                      # Water Marker (liquid is in S1 wl)
    102: "obj_monty_res",           # Monty Mole
    103: "obj_cheepcheep_res",      # Fish Bone -> Cheep Cheep (approx)
    104: "obj_angrysun_res",        # Angry Sun
    105: "obj_claw_res",            # Swinging Claw
    106: None,                      # Tree (decoration)
    107: None,                      # Piranha Creeper
    108: None,                      # Blinking Block
    109: None,                      # Sound Effect marker
    110: "obj_pinchos_res",         # Spike Block (approx)
    111: "obj_mechakoopa_res",      # Mechakoopa
    112: None,                      # Crate
    113: "obj_spring_res",          # Mushroom Trampoline (approx)
    114: None,                      # Porcupuffer
    115: None,                      # Cinobic
    116: None,                      # Super Hammer
    117: None,                      # Bully
    118: "obj_icicle_res",          # Icicle
    119: None,                      # ! Block
    120: "obj_ludwig_res",          # Lemmy   (approx -> only Ludwig in SMMWE)
    121: "obj_ludwig_res",          # Morton  (approx)
    122: "obj_ludwig_res",          # Larry   (approx)
    123: "obj_ludwig_res",          # Wendy   (approx)
    124: "obj_ludwig_res",          # Iggy    (approx)
    125: "obj_ludwig_res",          # Roy     (approx)
    126: "obj_ludwig_res",          # Ludwig
    127: None,                      # Cannon Box
    128: None,                      # Propeller Box
    129: "obj_cap_res",             # Goomba Mask (approx)
    130: "obj_cap_res",             # Bullet Bill Mask (approx)
    131: "obj_pow_res",             # Red POW Box (approx)
    132: "obj_spring_res",          # ON/OFF Trampoline (approx)
}

# Block `cid` (the MM2 id of the item inside) -> SMMWE S4 `sprout`.
# sprout=0 empty block, -1 multi-coin brick. Values verified in-game;
# unknown cids emit an empty block.
BLOCK_SPROUT_MAP = {
    8:   -1,   # Coin -> multi-coin block (from GML scr_edit_to_play)
    20:   1,   # Super Mushroom (verified)
    34:   2,   # Fire Flower (verified)
    44: -77,   # Style Power-up A / Cape / Mega Mario (verified)
    81: -85,   # Style Power-up B / Link (verified)
    # 33: ?,  # 1-Up Mushroom -- sprout value not yet verified
    # 35: ?,  # Super Star    -- sprout value not yet verified
}

# MM2 block ids that can contain items via cid link: Brick (4), Question (5), Hidden (29)
_BLOCK_IDS = {4, 5, 29}

# MM2 pipe direction lives in flag % 0x80.
MM2_PIPE_DIR = {0x00: "R", 0x20: "L", 0x40: "U", 0x60: "D"}

# SWE S5 pipe dir (0/1/2/3 = U/R/D/L) and rot, verified against four
# hand-placed pipes.
PIPE_DIR_MAP = {"U": 0, "R": 1, "D": 2, "L": 3}
PIPE_ROT = {0: 0, 1: -90, 2: 180, 3: -270}

# Skewer needs dir 1-4; dir=0 has no case in scr_edit_to_play and crashes on
# play. Flag bit 23 set = ceiling mount (dir=3/rot=180), clear = floor
# (dir=1/rot=0). Left/right mounts aren't mapped yet.
SKEWER_CEILING_FLAG = 1 << 23
SKEWER_DIR = {0: 1, SKEWER_CEILING_FLAG: 3}
SKEWER_ROT = {1: 0, 3: 180}

# Thwomps need dir=1 (face-down) to chase the player; dir=0 is inert.
# Up/left/right variants aren't mapped, so dir=1 is used for all.
THWOMP_DIR = 1

# One-Way Wall needs dir 1-4 (right/up/left/down); at dir=0 the loader hides
# the gate permanently (no default case in its switch). MM2 stores the facing
# in flag bits 22-23, mapped 1:1; the entry also needs rot = (dir-1)*90.
# Verified against one reference save per direction.
ONEWAY_DIR_MAP = {0: 1, 1: 2, 2: 3, 3: 4}

# Banzai Bill never launches with the generic S4 defaults. dir 1-4 =
# left/down/right/up from flag bits 22-23 (same field as One-Way Wall),
# rot = (dir-1)*90, scl = -1 for right/up. Verified against one reference
# save per direction.
BANZAI_DIR_MAP = {0: 1, 1: 2, 2: 3, 3: 4}
BANZAI_SCL = {1: 1, 2: 1, 3: -1, 4: -1}

# Flag bit 2 picks between two SWE variants: red/blue for ON/OFF (99) and
# Dotted-Line (100) blocks, regular/fire for the Clown Car (42). Tuple is
# (bit clear, bit set). Note 99 maps to obj_onoffblock_res and 100 to the
# onoffplatform pair. Verified against reference saves.
ONOFF_COLOR_FLAG = 1 << 2
ONOFF_SWE_IDS = {
    42:  ("obj_clown_res",           "obj_clown_fire_res"),
    99:  ("obj_onoffblock_res",      "obj_onoffblock_blue_res"),
    100: ("obj_onoffplatform_blue_res", "obj_onoffplatform_res"),
}

# Style Power-up A (id 44) is a different object per gamestyle. Only SMB1 and
# SMW have a usable mapping; SMB3/NSMBU are dropped rather than shown wrong.
STYLE_POWERUP_A_MAP = {
    0: "obj_mushroom_res",       # SMB1: Big Mushroom -> Mega Mario (best-effort fallback)
    2: "obj_cap_res",            # SMW: Cape Feather -> Cape Mario  (verified)
}

# Style Power-up B (id 81) shares one SMMWE slot in every gamestyle.
STYLE_POWERUP_B_MAP = {
    0: "obj_SMB2_mushroom_res",  # SMB1: Link (Master Sword)
    1: "obj_SMB2_mushroom_res",  # SMB3: Frog Suit
    2: "obj_SMB2_mushroom_res",  # SMW: Power Balloon
    3: "obj_SMB2_mushroom_res",  # NSMBU: Super Acorn
}

# Constant fields of an S5 pipe entry, from a hand-placed reference pipe.
S5_PIPE_TEMPLATE = {
    "sz": 0, "t_dir": 0, "clr": 0, "sclx": 1, "t_rot": 0, "wrp": 0,
    "t_s_sclx": 1, "msk": 0, "xscl": 1, "t_yscl": 1, "rot": 0, "t_sz": 0,
    "t_clr": 0, "yscl": 1, "t_xscl": 1, "dir": 0,
}

# Pixel delta subtracted from yy for objects whose SWE anchor sits one tile
# lower than the generic formula.
OBJ_Y_OFFSET_PX = {
    68: PX,   # Saw -> obj_grinder_res
    70: PX,   # Big Coin (10-coin) -> obj_coin10_res
}

# Ids anchored at the LEFT edge of the MM2 footprint (col = x // SUBPX)
# instead of the generic centered formula.
OBJ_LEFT_ANCHOR_IDS = {14, 16, 17, 31, 42, 49, 67, 71, 72}

# Ids whose single SWE sprite sits at the TOP of the h-tall MM2 footprint;
# yy is shifted up by (h-1) tiles so the sprite isn't buried in terrain.
# 13/14/16/47 get the same shift in their own S6/S7 builders.
OBJ_H_ANCHOR_TOP_IDS = {12, 32, 71}

# Flag bit 14 marks big-form enemies (bit 12 is wings, not big). SMMWE uses
# separate "_b_res" objects; only these four have working variants, and
# creating a nonexistent asset name crashes GameMaker, so every other enemy
# keeps its normal sprite. BIG_Y_OFFSET keeps the taller sprite on its row.
BIG_ENEMY_FLAG = 1 << 14
BIG_OBJ_ID_MAP = {
    "obj_goomba_res":    "obj_goomba_b_res",
    "obj_chomp_res":     "obj_chomp_b_res",
    "obj_koopa_res":     "obj_koopa_b_res",
    "obj_hammerbro_res": "obj_hammerbro_b_res",
}
BIG_Y_OFFSET = {
    "obj_goomba_res": PX,       # 1 tile taller
    "obj_chomp_res":  2 * PX,   # 2 tiles taller
}
_BIG_Y_DEFAULT = PX

# Every key an S4 entry carries; all flags default to 0. "dph" is required or
# scr_edit_to_play crashes in instance_create_depth on entering play mode.
S4_TEMPLATE = {
    "air": 0, "pinkcoin": 0, "fire": 0, "claw": 0, "key": 0, "rock": 0,
    "sprout": 0, "energy": 0, "wings": 0, "w_mode": 0, "rot": 0,
    "can_complement": 0, "parachute": 0, "progress": 0, "sierra": 0,
    "bumper": 0, "inup": 0, "ice": 0, "dph": 0,
}

# S6 cannon template from a hand-placed save (default placement had dwn=1).
# "dph" is required here too; same instance_create_depth crash as S4.
S6_CANNON_TEMPLATE = {
    "rht": 0, "dwn": 1, "up": 0, "lft": 0, "dir": 0, "dph": 0,
}

# Constant flags of an S7 "stretchy sprite" entry; contents/effects are not
# carried over.
S7_TEMPLATE = {
    "air": 0, "pinkcoin": 0, "clr": 0, "fire": 0, "key": 0, "rock": 0,
    "energy": 0, "wings": 0, "parachute": 0, "ice": 0,
}

# MM2 id -> S7 "ID" for the stretchy platform family. SMMWE swaps the names
# of 13 and 47: the Bullet Bill Blaster is "obj_bullebill_base_res" and the
# Cannon is "obj_cannon_res" (S6).
PLATFORM_S7_IDS = {
    16: "obj_semisolid_platform1",   # Semisolid Platform
    13: "obj_bullebill_base_res",    # Bullet Bill Blaster
    91: "obj_platform_res",          # Seesaw -> sized moving platform
    11: "obj_platform_res",          # Lift -> moving platform
    17: "obj_puente_res",            # Bridge
    49: "obj_puente_res",            # Castle Bridge (approx)
    14: "obj_mushroom_platform_res", # Mushroom Platform
}

# Bridges always use this sprite; the reference save used it regardless of
# theme and no variants are confirmed.
PUENTE_SPRITE = "spr_NSMBU_puente_underground"

# SWE gamestyle -> sprite-name prefix (SMW has none).
GAMESTYLE_SPR_PREFIX = {0: "SMB", 1: "SMB3", 2: "", 3: "NSMBU"}

# Themes with a spr_<prefix>_ssp1_<theme> sprite in data.win; anything else
# falls back to overworld.
SSP1_THEMES = {
    "SMB":   {"airship", "castle", "desert", "ghost", "overworld", "snow",
              "underground", "underwater"},
    "SMB3":  {"airship", "castle", "desert", "forest", "ghost", "overworld",
              "sky", "snow", "underground", "underwater"},
    "":      {"airship", "castle", "desert", "forest", "ghost", "overworld",
              "snow", "underground", "underwater"},
    "NSMBU": {"airship", "castle", "desert", "forest", "ghost", "overworld",
              "sky", "snow", "underground", "underwater"},
}


def ssp_sprite_name(gamestyle, theme):
    """Semisolid Platform sprite name for a given SWE gamestyle/theme."""
    prefix = GAMESTYLE_SPR_PREFIX.get(gamestyle, "NSMBU")
    valid = SSP1_THEMES.get(prefix, SSP1_THEMES["NSMBU"])
    t = theme if theme in valid else "overworld"
    return f"spr_{prefix}_ssp1_{t}" if prefix else f"spr_ssp1_{t}"


# The mp1 (Mushroom Platform) sprite set is sparse: SMB1/SMB3/SMW only ship
# airship/snow/underwater variants, NSMBU everything except sky. A sprite
# name that doesn't exist crashes obj_modelsizable, hence the fallbacks.
MP1_THEMES = {
    "":      {"airship", "snow", "underwater"},
    "SMB":   {"airship", "snow", "underwater"},
    "SMB3":  {"airship", "snow", "underwater"},
    "NSMBU": {"airship", "castle", "desert", "forest", "ghost", "overworld",
              "snow", "underground", "underwater"},
}
# Fallback when the theme has no mp1 variant.
MP1_FALLBACK = {
    "":      "spr_mp1",
    "SMB":   "spr_SMB_mp1",
    "SMB3":  "spr_SMB3_mp1",
    "NSMBU": "spr_NSMBU_mp1_overworld",
}


def mp1_sprite_name(gamestyle, theme):
    """Mushroom Platform sprite for this gamestyle/theme (see MP1_THEMES)."""
    prefix = GAMESTYLE_SPR_PREFIX.get(gamestyle, "NSMBU")
    valid = MP1_THEMES.get(prefix, MP1_THEMES["NSMBU"])
    if theme in valid:
        return f"spr_{prefix}_mp1_{theme}" if prefix else f"spr_mp1_{theme}"
    return MP1_FALLBACK.get(prefix, MP1_FALLBACK["NSMBU"])


def bullebill_sprite_name(gamestyle):
    """Blaster sprite name; only SMB3/NSMBU have a dedicated one."""
    prefix = GAMESTYLE_SPR_PREFIX.get(gamestyle, "NSMBU")
    if prefix in ("SMB3", "NSMBU"):
        return f"spr_{prefix}_bullebill_base"
    return "spr_bullebill_base"


def platform_sprite_name(gamestyle):
    """Moving platform sprite name for a SWE gamestyle."""
    prefix = GAMESTYLE_SPR_PREFIX.get(gamestyle, "NSMBU")
    return f"spr_{prefix}_platform" if prefix else "spr_platform"


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def ground_xx(tile_x):
    return tile_x * PX


def ground_yy(tile_y):
    """tile_y is rows-from-bottom; SWE yy is pixels-from-top."""
    return (FIELD_HEIGHT_TILES - 1 - tile_y) * PX


def marker_yy(tile_y):
    """Start/goal markers sit one row lower than ground tiles."""
    return (FIELD_HEIGHT_TILES - tile_y) * PX


def object_cell(o):
    """(col, row from bottom) of an MM2 object: col = x//160 - w//2 and
    row = y//160; OBJ_LEFT_ANCHOR_IDS use col = x//160."""
    w = max(1, o.get("w", 1))
    if o.get("id") in OBJ_LEFT_ANCHOR_IDS:
        col = o["x"] // SUBPX
    else:
        col = o["x"] // SUBPX - w // 2
    row = o["y"] // SUBPX
    return col, row


# Ids whose bounding box isn't solid terrain; excluded from occupied_cells.
_NON_SOLID_IDS = {87, 88, 16}  # slopes, semisolid platform


def occupied_cells(j, *, exclude_id=None, extra_ground=None):
    """Cells covered by ground or object footprints (skipping exclude_id and
    _NON_SOLID_IDS), plus extra_ground. Used by build_cannons."""
    occ = {(g["x"], g["y"]) for g in j.get("ground", [])}
    if extra_ground:
        occ |= extra_ground
    for o in j.get("objects", []):
        oid = o.get("id")
        if oid == exclude_id or oid in _NON_SOLID_IDS:
            continue
        w = max(1, o.get("w", 1))
        h = max(1, o.get("h", 1))
        col, row = object_cell(o)
        for dx in range(w):
            for dy in range(h):
                occ.add((col + dx, row + dy))
    return occ


def skewer_footprint_cells(objects):
    """Cells covered by each Skewer's bounding box. SMMWE won't let a skewer
    overlap S2 ground, so build_world drops these cells from S2."""
    cells = set()
    for o in objects:
        if o.get("id") != 83:
            continue
        col, row = object_cell(o)
        w = max(1, o.get("w", 1))
        h = max(1, o.get("h", 1))
        for dx in range(w):
            for dy in range(h):
                cells.add((col + dx, row + dy))
    return cells


# Slope id -> run per 1-tile rise: Steep (88) rises every column, Slight (87)
# every 2. SWE has no slopes; build_world fills them in as stepped ground.
_SLOPE_STEPS = {88: 1, 87: 2}
_SLOPE_IDS = set(_SLOPE_STEPS)


def slope_fill_cells(objects):
    """Staircase of ground cells for every slope object, merged into S2.
    Each of the w columns gets a step of ceil(run/step) tiles capped at h;
    flag & 0x100000 flips which end is low. Slopes anchor at their left
    edge. Verified against 3000065_1_overworld.json."""
    cells = set()
    for o in objects:
        step = _SLOPE_STEPS.get(o.get("id"))
        if step is None:
            continue
        w = max(1, o.get("w", 1))
        h = max(1, o.get("h", 1))
        base_col = o["x"] // SUBPX
        base_row = o["y"] // SUBPX
        descending = (o.get("flag", 0) & 0x100000) != 0
        for x in range(w):
            run = (w - x) if descending else (x + 1)
            height = min((run + step - 1) // step, h)  # ceil(run/step), capped at h
            for y in range(height):
                cells.add((base_col + x, base_row + y))
    return cells


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def build_ground(ground):
    """MM2 ground[] -> SWE S2, recomputing the autotile index from 4-neighbour
    occupancy."""
    occ = {(g["x"], g["y"]) for g in ground}
    out = []
    for g in ground:
        x, y = g["x"], g["y"]
        mask = (
            ((x, y + 1) in occ) << 0   # N: a higher row (y+1) is "above"
            | ((x + 1, y) in occ) << 1  # E
            | ((x, y - 1) in occ) << 2  # S: lower row (y-1) is "below"
            | ((x - 1, y) in occ) << 3  # W
        )
        out.append({"xx": ground_xx(x), "yy": ground_yy(y), "i": GROUND_AUTOTILE[mask]})
    return out


def build_pipes(objects):
    """MM2 pipes (id 9) -> SWE S5 entries. Returns (s5_list, consumed_plants),
    where consumed_plants holds id()s of Piranha Plants folded into a pipe
    (msk=1) that build_objects must not emit as S4 objects.

    From four hand-placed reference pipes: dir = 0/1/2/3 for U/R/D/L,
    rot = -90*dir, sclx = -1 for D/L, sz = length - 2, the length axis is
    yscl for vertical and xscl for horizontal, t_x_pos = xx + 32, and
    t_y_pos = yy + 32 for U only. (xx, yy) is the top-left corner of the
    pipe's tile footprint; only the U anchor is in-game verified."""
    # Piranha plant (id=2) lookup by tile cell for msk=1 detection.
    plant_by_cell = {}
    for o in objects:
        if o.get("id") == 2:
            plant_by_cell.setdefault((o["x"] // SUBPX, o["y"] // SUBPX), o)

    out = []
    consumed_plants = set()
    for o in objects:
        if o.get("id") != 9:
            continue
        length = max(1, o.get("h", 1))
        base_col = o["x"] // SUBPX
        base_row = o["y"] // SUBPX
        direction = MM2_PIPE_DIR.get(o.get("flag", 0) % 0x80, "R")

        if direction == "U":
            col, row_top = base_col, base_row + length - 1
        elif direction == "D":
            col, row_top = base_col - 1, base_row
        elif direction == "R":
            col, row_top = base_col, base_row
        else:  # L
            col, row_top = base_col - length + 1, base_row + 1

        xx = col * PX
        yy = ground_yy(row_top)
        dir_idx = PIPE_DIR_MAP[direction]

        entry = dict(S5_PIPE_TEMPLATE)
        if dir_idx % 2 == 0:   # U, D
            entry["yscl"] = length / 2
        else:                  # R, L
            entry["xscl"] = length / 2
        entry.update({
            "sz": length - 2,
            "sclx": -1 if dir_idx >= 2 else 1,
            "rot": PIPE_ROT[dir_idx],
            "dir": dir_idx,
            "xx": xx, "yy": yy,
            "t_x_pos": xx + 2 * PX,
            "t_y_pos": yy + 2 * PX if direction == "U" else yy,
        })
        # A plant at or one tile above an upward pipe's mouth becomes msk=1
        # instead of a separate S4 object.
        if direction == "U":
            for dc in (0, 1):
                for dr in (0, 1):
                    plant = plant_by_cell.get((base_col + dc, row_top + dr))
                    if plant is not None:
                        entry["msk"] = 1
                        consumed_plants.add(id(plant))
        out.append(entry)
    return out, consumed_plants


# MM2 object ids handled by dedicated builders, not the generic S4 path.
_NON_S4_IDS = {9, 11, 13, 14, 16, 17, 47, 49, 55, 91}  # 9=Pipe (S5), 55=Door (S8), 11/13/14/16/17/47/49/91 -> S6/S7 (see below)


def build_objects(objects, gamestyle=3, consumed_plants=None):
    """MM2 objects[] -> SWE S4. Returns (s4_list, dropped_counts). Ids with
    dedicated builders (_NON_S4_IDS, slopes) are skipped; consumed_plants
    are plants already folded into pipes."""
    # A block's cid stores the contained item's MM2 id directly; map it to a
    # sprout value.
    block_sprouts = {}  # id(block_obj) -> sprout int
    for o in objects:
        if o.get("id") in _BLOCK_IDS:
            sprout = BLOCK_SPROUT_MAP.get(o.get("cid", -1))
            if sprout is not None:
                block_sprouts[id(o)] = sprout

    out = []
    dropped = {}
    for o in objects:
        oid = o.get("id")
        if oid in _NON_S4_IDS or oid in _SLOPE_IDS:
            continue
        if consumed_plants and oid == 2 and id(o) in consumed_plants:
            continue
        if oid == 44:
            swe_id = STYLE_POWERUP_A_MAP.get(gamestyle)
        elif oid == 81:
            swe_id = STYLE_POWERUP_B_MAP.get(gamestyle)
        elif oid in ONOFF_SWE_IDS:
            off_id, on_id = ONOFF_SWE_IDS[oid]
            swe_id = on_id if o.get("flag", 0) & ONOFF_COLOR_FLAG else off_id
        else:
            swe_id = OBJ_ID_MAP.get(oid)
        if swe_id is None:
            name = o.get("name", f"id={oid}")
            dropped[name] = dropped.get(name, 0) + 1
            continue
        col, row = object_cell(o)
        scl = 1
        yy = (FIELD_HEIGHT_TILES - 1 - row) * PX - OBJ_Y_OFFSET_PX.get(oid, 0)
        if oid in OBJ_H_ANCHOR_TOP_IDS:
            h = max(1, o.get("h", 1))
            yy -= (h - 1) * PX
        if o.get("flag", 0) & BIG_ENEMY_FLAG and swe_id in BIG_OBJ_ID_MAP:
            yy -= BIG_Y_OFFSET.get(swe_id, _BIG_Y_DEFAULT)
            swe_id = BIG_OBJ_ID_MAP[swe_id]
        entry = dict(S4_TEMPLATE)
        entry.update({
            "ID": swe_id,
            "xx": col * PX,
            "yy": yy,
            "scl": scl,
            "dir": 0,   # best-effort: MM2 flag bitfields aren't carried over
        })
        if oid == 12:
            entry["dir"] = THWOMP_DIR
        elif oid == 67:
            entry["dir"] = ONEWAY_DIR_MAP.get((o.get("flag", 0) >> 22) & 0x3, 1)
            entry["rot"] = (entry["dir"] - 1) * 90
        elif oid == 32:
            entry["dir"] = BANZAI_DIR_MAP.get((o.get("flag", 0) >> 22) & 0x3, 1)
            entry["rot"] = (entry["dir"] - 1) * 90
            entry["scl"] = BANZAI_SCL[entry["dir"]]
        elif oid == 83:
            # Skewers anchor one column right and one row above the generic
            # cell (from hand-placed reference skewers).
            w = max(1, o.get("w", 1))
            entry["xx"] = (col + w // 2) * PX
            entry["yy"] = yy - PX
            entry["dir"] = SKEWER_DIR.get(o.get("flag", 0) & SKEWER_CEILING_FLAG, 1)
            entry["rot"] = SKEWER_ROT.get(entry["dir"], 0)
        if id(o) in block_sprouts:
            entry["sprout"] = block_sprouts[id(o)]
        out.append(entry)
    return out, dropped


def build_doors(objects):
    """MM2 doors (id 55) -> SWE S8 "obj_door_res" entries. SMMWE stores a
    door PAIR per entry: xx/yy for one door, dr_x/dr_y for its partner.
    Flag bits 19-22 hold the pairing order; sort by it and pair
    consecutively, the higher-order door taking xx/yy. A trailing unpaired
    door is dropped."""
    doors = sorted(
        (o for o in objects if o.get("id") == 55),
        key=lambda o: (o.get("flag", 0) >> 19) & 0xF,
    )
    out = []
    for lo, hi in zip(doors[0::2], doors[1::2]):
        c_lo, r_lo = object_cell(lo)
        c_hi, r_hi = object_cell(hi)
        out.append({
            "xx": c_hi * PX,
            "yy": (FIELD_HEIGHT_TILES - 1 - r_hi) * PX,
            "dr_x": c_lo * PX,
            "dr_y": (FIELD_HEIGHT_TILES - 1 - r_lo) * PX,
            "ID": "obj_door_res",
        })
    return out


def build_cannons(objects, occ):
    """MM2 Cannon (id 47) -> SWE S6 "obj_cannon_res" entries.

    The S6 entry has no size field, so MM2's h is lost. Mount and direction
    are inferred from adjacent terrain in `occ`: solid above = ceiling mount
    (up=1); a ceiling cannon with open space below shoots down (dir=6),
    otherwise it faces away from a side wall. Floor cannons face right.
    Flag bit 26 clear flips left/right. A big (h>1) floor cannon's tile goes
    on the open row above its embedded box. Verified against the 21 cannons
    in a real level plus a reference save."""
    HFLIP_DIR = {3: 7, 7: 3}   # dir 3 = right, 7 = left
    out = []
    for o in objects:
        if o.get("id") != 47:
            continue
        w = max(1, o.get("w", 1))
        h = max(1, o.get("h", 1))
        col, row = object_cell(o)

        ceiling = any((col + dx, row + h) in occ for dx in range(w))
        if ceiling:
            up, dwn = 1, 0
            below_open = not any((col + dx, row - 1) in occ for dx in range(w))
            if below_open:
                cannon_dir = 6
            else:
                wall_left = any((col - 1, row + dy) in occ for dy in range(h))
                wall_right = any((col + w, row + dy) in occ for dy in range(h))
                cannon_dir = 7 if (wall_right and not wall_left) else 3
            top_row = row + h - 1
        else:
            up, dwn = 0, 1
            cannon_dir = 3
            top_row = row + h if h > 1 else row

        if not (o.get("flag", 0) & (1 << 26)):
            cannon_dir = HFLIP_DIR.get(cannon_dir, cannon_dir)

        entry = dict(S6_CANNON_TEMPLATE)
        entry.update({
            "ID": "obj_cannon_res",
            "xx": col * PX,
            "yy": ground_yy(top_row),
            "up": up,
            "dwn": dwn,
            "dir": cannon_dir,
        })
        out.append(entry)
    return out


def repair_semisolid_objects(objects, ground):
    """Rebuild broken Semisolid Platform (id 16) objects into clean boxes via
    repair_semisolid_cells, for JSONs that still carry the diffusion model's
    scattered fragments. Each object is repaired independently and the pass
    is idempotent; everything else passes through untouched."""
    ground_cells = {(g.get("x"), g.get("y")) for g in ground}
    rest = []
    for o in objects:
        if o.get("id") != 16:
            rest.append(o)
            continue
        col, row = object_cell(o)
        w = max(1, o.get("w", 1))
        h = max(1, o.get("h", 1))
        cells = {(col + dx, row + dy) for dx in range(w) for dy in range(h)}
        for c0, r0, bw, bh in repair_semisolid_cells(cells, ground_cells):
            # id 16 is left-anchored (col = x // SUBPX)
            rest.append({"id": 16, "name": "Semisolid Platform",
                         "x": c0 * SUBPX, "y": r0 * SUBPX, "w": bw, "h": bh,
                         "flag": 0, "cflag": 0})
    return rest


def build_platform_objects(objects, *, gamestyle, theme):
    """Blaster (13), Semisolid (16), Seesaw (91), Lift (11), Bridge (17/49)
    and Mushroom Platform (14) -> SWE S7 "stretchy sprite" entries.

    `spr` picks the themed sprite, wth/hht size it in tiles (w/h directly;
    verified for semisolids and mushroom platforms). Seesaw and Lift map to
    the moving platform obj_platform_res with dph=-1/dir=2 (routing Lift
    through S4 crashed on the missing dph/wth/hht/spr fields). Bridges use
    PUENTE_SPRITE with hht=3/dph=8 and sit 2 tiles above the generic
    anchor, as does the seesaw. Mushroom Platforms use dph = 245 + hht."""
    out = []
    for o in objects:
        oid = o.get("id")
        if oid not in PLATFORM_S7_IDS:
            continue
        w = max(1, o.get("w", 1))
        h = max(1, o.get("h", 1))
        col, row = object_cell(o)
        if oid == 91:
            wth, hht = max(4, w), 3
        elif oid == 11:
            wth, hht = max(4, w), 1
        elif oid in (17, 49):
            wth, hht = w, 3
        else:
            wth, hht = w, h
        yy = ground_yy(row) - (hht - 1) * PX
        if oid == 91:
            yy += 2 * PX  # nudge the seesaw replacement down 2 tiles
        elif oid in (17, 49):
            yy += 2 * PX  # bridge sprite sits 2 tiles above the generic anchor
        entry = dict(S7_TEMPLATE)
        entry.update({
            "ID": PLATFORM_S7_IDS[oid],
            "xx": col * PX,
            "yy": yy,
            "dir": 0,
            "wth": wth, "hht": hht,
        })
        if oid == 16:
            entry.update({"spr": ssp_sprite_name(gamestyle, theme), "dph": 255})
        elif oid == 13:
            entry.update({"spr": bullebill_sprite_name(gamestyle), "dph": 0})
        elif oid in (17, 49):
            entry.update({"spr": PUENTE_SPRITE, "dph": 8})
        elif oid == 14:
            entry.update({"spr": mp1_sprite_name(gamestyle, theme), "dph": 245 + hht})
        else:  # 91 (Seesaw) / 11 (Lift) -- moving platform
            entry.update({"spr": platform_sprite_name(gamestyle), "dph": -1, "dir": 2})
        out.append(entry)
    return out


def build_metadata(j, *, user, name, desc, date_str, time_str):
    """MM2 level header -> SWE S1 metadata dict."""
    gamestyle_raw = j.get("gamestyle_raw", 0)
    theme_raw = j.get("theme_raw", 0)

    goal_x_tenths = j.get("goal_x", 0)            # tenths of a tile
    goal_col = goal_x_tenths // 10
    goal_y_tiles = j.get("goal_y", 0)             # tiles from bottom
    start_y_tiles = j.get("start_y", 0)           # tiles from bottom

    # Level width in pixels: prefer the stored boundary, else span the content.
    right = j.get("right_boundary", 0)
    if right > 0:
        size = right
    else:
        cols = [g["x"] for g in j.get("ground", [])]
        cols += [o["x"] // SUBPX for o in j.get("objects", [])]
        size = (max(cols) + 2) * PX if cols else FIELD_HEIGHT_TILES * PX

    # Liquid / water level.
    liquid_start = j.get("liquid_start_height", 0)
    if liquid_start:
        wl = (FIELD_HEIGHT_TILES - liquid_start) * PX
    else:
        wl = (FIELD_HEIGHT_TILES - 1) * PX
    wl_speed_map = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6}
    wl_speed = wl_speed_map.get(j.get("liquid_speed_raw", 0), 0.0)

    return {
        "goal_y": marker_yy(goal_y_tiles),
        "nightmode": 1 if j.get("night_time") else 0,
        "label_1": 0,
        "autoscroll": 1 if j.get("autoscroll_type_raw", 0) else 0,
        "t_conditions": 0,
        "wl": wl,
        "mtrs": 0,
        "wl_speed": wl_speed,
        "c_conditions": 0,
        "conditions": 0,
        "t": 1,
        "start_y": marker_yy(start_y_tiles),
        "date": date_str,
        "goal_x": goal_col * PX,
        "size": size,
        "gamestyle": GAMESTYLE_MAP.get(gamestyle_raw, 3),
        "ds_s": 0,
        "desc": desc if desc is not None else (j.get("description", "") or ""),
        "label_2": -1,
        "user": user,
        "time": time_str,
        "gametheme": THEME_MAP.get(theme_raw, "overworld"),
        "wl_limit": wl,
        "o_conditions": 0,
        "timer": j.get("timer", 0),
    }


def build_world(j, *, user, name, desc, date_str, time_str):
    """Build a full SWE world dict (S0 or a populated SB1) from one map JSON."""
    objects = repair_semisolid_objects(j.get("objects", []), j.get("ground", []))
    gamestyle = GAMESTYLE_MAP.get(j.get("gamestyle_raw", 0), 3)
    theme = THEME_MAP.get(j.get("theme_raw", 0), "overworld")
    s5, consumed_plants = build_pipes(objects)
    s4, dropped = build_objects(objects, gamestyle, consumed_plants)
    s8 = build_doors(objects)

    # Slopes have no SWE equivalent; fill them in as solid ground.
    slope_cells = slope_fill_cells(objects)
    ground = list(j.get("ground", []))
    existing_ground = {(g["x"], g["y"]) for g in ground}
    for x, y in slope_cells:
        if (x, y) not in existing_ground:
            ground.append({"x": x, "y": y})
            existing_ground.add((x, y))

    occ = occupied_cells(j, exclude_id=47, extra_ground=slope_cells)
    s6 = build_cannons(objects, occ)
    s7 = build_platform_objects(objects, gamestyle=gamestyle, theme=theme)

    # Drop ground under skewers; SMMWE forbids the overlap.
    skewer_cells = skewer_footprint_cells(objects)
    ground = [g for g in ground if (g["x"], g["y"]) not in skewer_cells]

    world = {
        "S1": [build_metadata(j, user=user, name=name, desc=desc,
                              date_str=date_str, time_str=time_str)],
        "S2": build_ground(ground),
        "S3": [],
        "S4": s4,
        "S5": s5,
        "S6": s6,
        "S7": s7,
        "S8": s8,
    }
    return world, dropped


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_swe(level_dict):
    """level_dict -> .swe bytes: base64 JSON + HMAC-SHA1 hex + NUL.

    The editor NUL-terminates the string (GameMaker buffer_string); without
    the terminator the level silently fails to load."""
    payload = json.dumps(level_dict, separators=(",", ":"), ensure_ascii=False)
    b64 = base64.b64encode(payload.encode("utf-8"))
    checksum = hmac.new(SWE_HMAC_KEY, b64, hashlib.sha1).hexdigest()
    return b64 + checksum.encode("ascii") + b"\x00"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_companion(path: Path):
    """Given an *_overworld.json / *_subworld.json, return
    (overworld_path, subworld_path, base_stem) with missing files as None."""
    stem = path.stem
    if stem.endswith("_overworld"):
        base = stem[: -len("_overworld")]
        ow, sub = path, path.with_name(base + "_subworld" + path.suffix)
    elif stem.endswith("_subworld"):
        base = stem[: -len("_subworld")]
        ow, sub = path.with_name(base + "_overworld" + path.suffix), path
    else:
        base = stem
        ow, sub = path, None
    ow = ow if ow and ow.exists() else None
    sub = sub if sub and sub.exists() else None
    return ow, sub, base


def detect_smmwe_user(default="patwick"):
    """Username SMMWE is logged in as, read from its Settings*.dat (the lone
    identifier-looking line). SMMWE filters levels by author, so exports
    should carry the player's own name. Falls back to `default`."""
    import os
    import re
    import glob
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        return default
    for path in sorted(glob.glob(os.path.join(base, "SMM_WE", "Settings*.dat"))):
        try:
            text = open(path, encoding="latin1").read()
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", line):
                return line
    return default


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert an MM2 level JSON (or a folder of JSONs) into "
                    "Super Mario Maker Worldwide Engine (.swe) save file(s)."
    )
    p.add_argument("--input", required=True, help="Path to a *_overworld.json, a plain level JSON, or a folder of JSONs")
    p.add_argument("-o", "--output_folder", help="Output .swe path (single-file) or output folder (folder mode)")
    p.add_argument("--user", default=None,
                   help="Author name stored in the level (default: the SMMWE "
                        "logged-in username, else 'patwick')")
    p.add_argument("--name", default=None, help="Level name (default: JSON 'name'); ignored in folder mode")
    p.add_argument("--desc", default=None, help="Description (default: JSON 'description')")
    p.add_argument("--height", type=int, default=FIELD_HEIGHT_TILES,
                   help=f"Playfield height in tiles (default {FIELD_HEIGHT_TILES})")
    return p.parse_args(argv)


def convert_one(ow_path, sub_path, base, args, out_dir=None):
    """Convert one overworld+optional subworld pair to a .swe file."""
    overworld_json = load_json(ow_path)
    subworld_json = load_json(sub_path) if sub_path else None

    now = datetime.now()
    date_str = now.strftime("%d/%m/%Y")
    time_str = now.strftime("%H:%M")
    name = args.name if args.name is not None else overworld_json.get("name", base)

    s0, dropped = build_world(overworld_json, user=args.user, name=name,
                              desc=args.desc, date_str=date_str, time_str=time_str)

    if subworld_json is not None:
        sb1, sub_dropped = build_world(subworld_json, user=args.user, name=name,
                                       desc=args.desc, date_str=date_str, time_str=time_str)
        for k, v in sub_dropped.items():
            dropped[k] = dropped.get(k, 0) + v
    else:
        sb1 = {"S1": []}

    level = {"S0": s0, "SB1": sb1}
    data = encode_swe(level)

    if out_dir is not None:
        out_path = Path(out_dir) / (base + ".swe")
    elif args.output_folder:
        out_path = Path(args.output_folder)
    else:
        out_path = ow_path.with_name(base + ".swe")
    out_path.write_bytes(data)

    print(f"Wrote {out_path} ({len(data)} bytes)")
    print(f"  ground tiles : {len(s0['S2'])}")
    print(f"  objects      : {len(s0['S4'])}")
    print(f"  pipes        : {len(s0['S5'])}")
    print(f"  cannons      : {len(s0['S6'])}")
    print(f"  platforms    : {len(s0['S7'])}")
    print(f"  doors        : {len(s0['S8'])}")
    if dropped:
        total = sum(dropped.values())
        print(f"  dropped {total} object(s) with no SMMWE equivalent:")
        for name_, n in sorted(dropped.items(), key=lambda kv: -kv[1]):
            print(f"      {n:4d}  {name_}")


def main(argv=None):
    global FIELD_HEIGHT_TILES
    args = parse_args(argv)
    FIELD_HEIGHT_TILES = args.height
    if args.user is None:
        args.user = detect_smmwe_user()
        print(f"Author: {args.user}")

    in_path = Path(args.input)

    if in_path.is_dir():
        out_dir = Path(args.output_folder) if args.output_folder else in_path
        out_dir.mkdir(parents=True, exist_ok=True)
        seen_bases = set()
        errors = 0
        for json_file in sorted(in_path.glob("*.json")):
            if json_file.stem.endswith("_subworld"):
                continue  # picked up as companion of overworld
            ow_path, sub_path, base = find_companion(json_file)
            if base in seen_bases:
                continue
            seen_bases.add(base)
            if ow_path is None:
                print(f"Skipping {json_file.name}: no overworld JSON found")
                continue
            try:
                convert_one(ow_path, sub_path, base, args, out_dir=out_dir)
            except Exception as exc:
                print(f"ERROR converting {json_file.name}: {exc}")
                errors += 1
        print(f"\nDone: {len(seen_bases)} level(s) processed, {errors} error(s).")
    else:
        ow_path, sub_path, base = find_companion(in_path)
        if ow_path is None and sub_path is None:
            raise SystemExit(f"Could not find JSON data at {in_path}")
        convert_one(ow_path, sub_path, base, args)


if __name__ == "__main__":
    main()
