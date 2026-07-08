import os
import re
import json
from PIL import Image
import util.common_settings as common_settings

# ---------------------------------------------------------------------------
# Mario Maker 2 tiles
#
# Unlike the SMB/LR/MM sheets above (a fixed grid of equal tiles indexed by
# hardcoded (row, col)), MM2 sprites are arbitrary {x, y, w, h} rectangles
# packed into img/spritesheet.png. To turn a mm2_tileset_we.json glyph into a
# sprite we chain three tables:
#
#   glyph  --OBJ_META-->  object name  --NAME_TO_ID-->  object id
#   (object id, gamestyle)  --_MM2_SPRITE_DATA-->  (x, y, w, h)  in spritesheet
#
# The rectangles in _MM2_SPRITE_DATA are baked from toost's LevelData.hpp
# ObjectLocation map (transcribed offline, not parsed at runtime) so the ascii
# browser ships everything it needs -- it renders real MM2 art with only the
# committed spritesheet/tilesheet PNGs, no external toost checkout required.
# ---------------------------------------------------------------------------

# Cache so the spritesheet PNG is read only once.
_mm2_spritesheet = None    # PIL RGBA image of img/spritesheet.png
_mm2_tilesheet_cache = {}  # {(gamestyle, theme): PIL RGBA of img/tile/<raw>-<theme>.png}

# Gamestyles tried in order when the requested style lacks a sprite for an
# object (e.g. style-exclusive items). SMW first because that is the pipeline
# default (mm2pipeline_data.ascii to-json --gamestyle smw).
_MM2_STYLE_FALLBACK = ("SMW", "SMB1", "SMB3", "NSMBU", "SM3DW")
_MM2_GAMESTYLE_RAW = {
    "SMB1": 12621, "SMB3": 13133, "SMW": 22349, "NSMBU": 21847, "SM3DW": 22323,
}

# Tile-type objects (blocks, coins, spikes, ...) are NOT in the spritesheet --
# toost draws them from the gamestyle/theme tilesheet (img/tile/<raw>-<theme>.png)
# via DrawTile. These (x, y) 16px cells come from LevelDrawer::Setup's TileLoc
# table plus the representative cell of a few autotiled objects (pipe 9, mushroom
# platform 14, semisolid 16, bridge 17, vine 64). For the single-cell per-tile
# path: the multi-tile ones (pipe/mushroom/semisolid/bridge) get proper edge tiles
# from _mm2_autotile, and fall back to this one cell only when that can't run.
_MM2_TILESHEET_CELL = {
    4:  (1, 0),    # Block
    5:  (2, 0),    # ? Block
    6:  (6, 0),    # Hard Block
    8:  (7, 0),    # Coin
    9:  (12, 0),   # Pipe (body cell; pipes are multi-tile)
    14: (4, 2),    # Mushroom Platform (cap centre)
    16: (8, 3),    # Semisolid Platform (top surface)
    17: (1, 2),    # Bridge
    21: (0, 4),    # Donut Block
    22: (6, 6),    # Cloud
    23: (4, 0),    # Note Block
    29: (3, 0),    # Hidden Block
    43: (2, 4),    # Spikes
    63: (8, 7),    # Ice Block
    64: (14, 7),   # Vine (middle segment)
    99: (2, 23),   # ON/OFF Block
    100: (3, 22),  # Dotted-Line Block
}
# Fully-surrounded interior ground cell for the '#' glyph: GrdLoc[255] from
# LevelDrawer::Setup's GS[] table (GS[255] = 0x6F -> X=6, Y=15).
_MM2_GROUND_CELL = (6, 15)
# Spritesheet rectangles (x, y, w, h in img/spritesheet.png) for every MM2 object
# id the tileset can reference, per gamestyle. Transcribed from toost's
# LevelData.hpp ObjectLocation map (the authoritative source) so the renderer no
# longer needs that 2900-line C++ header at runtime -- it ships the data it needs.
# Only enemies/items/bosses/platforms live here; pure tile objects (blocks, coins,
# ground, pipes, ...) are drawn from the tilesheet (_MM2_TILESHEET_CELL / autotile)
# and are intentionally absent. Piranha Plant (id 2) uses toost's OBJ_2A0 variant
# (down-facing baseline form). Keep in sync with mm2_tileset_we.json's glyphs.
_MM2_SPRITE_DATA = {
    2: {"SMB1": (1168,488,32,16), "SMB3": (1184,856,32,16), "SMW": (896,1168,32,16), "NSMBU": (1136,1120,32,16), "SM3DW": (576,1168,32,16)},
    0: {"SMB1": (1216,1184,16,16), "SMB3": (656,1232,16,16), "SMW": (1312,1232,16,16), "NSMBU": (1264,736,16,16), "SM3DW": (848,1280,16,16)},
    1: {"SMB1": (1200,272,16,32), "SMB3": (1184,776,16,32), "SMW": (32,1200,16,32), "NSMBU": (80,1168,16,32), "SM3DW": (1072,704,16,64)},
    3: {"SMB1": (1200,552,16,32), "SMB3": (1184,952,16,32), "SMW": (192,1200,16,32), "NSMBU": (240,1168,16,32), "SM3DW": (960,288,16,64)},
    10: {"SMB1": (1200,408,16,16), "SMB3": (688,1232,16,16), "SMW": (1312,1264,16,16), "NSMBU": (1264,768,16,16), "SM3DW": (880,1280,16,16)},
    11: {"SMB1": (1200,936,16,16), "SMB3": (768,1232,16,16), "SMW": (32,1312,16,16), "NSMBU": (1264,848,16,16), "SM3DW": (976,1280,16,16)},
    12: {"SMB1": (976,816,32,32), "SMB3": (736,976,32,32), "SMW": (960,1104,32,32), "NSMBU": (1040,544,32,32), "SM3DW": (352,1072,32,32)},
    13: {"SMB1": (1200,424,16,32), "SMB3": (1184,824,16,32), "SMW": (80,1200,16,32), "NSMBU": (128,1168,16,32), "SM3DW": (1216,288,16,32)},
    15: {"SMB1": (704,1184,16,16), "SMB3": (1216,1232,16,16), "SMW": (480,1312,16,16), "NSMBU": (32,1264,16,16), "SM3DW": (1296,128,16,16)},
    18: {"SMB1": (736,1184,16,16), "SMB3": (1248,0,16,16), "SMW": (512,1312,16,16), "NSMBU": (64,1264,16,16), "SM3DW": (1296,160,16,16)},
    19: {"SMB1": (784,1184,16,16), "SMB3": (1248,48,16,16), "SMW": (560,1312,16,16), "NSMBU": (112,1264,16,16), "SM3DW": (1296,208,16,16)},
    20: {"SMB1": (816,1184,16,16), "SMB3": (1248,80,16,16), "SMW": (592,1312,16,16), "NSMBU": (144,1264,16,16), "SM3DW": (1296,240,16,16)},
    24: {"SMB1": (848,1184,16,16), "SMB3": (1248,112,16,16), "SMW": (624,1312,16,16), "NSMBU": (176,1264,16,16), "SM3DW": (1296,272,16,16)},
    25: {"SMB1": (864,1184,16,16), "SMB3": (1248,128,16,16), "SMW": (640,1312,16,16), "NSMBU": (192,1264,16,16), "SM3DW": (1296,288,16,16)},
    27: {"SMB1": (864,240,16,176), "SMB3": (1008,0,32,32), "SMW": (864,1168,32,16), "NSMBU": (880,240,16,176), "SM3DW": (896,240,16,176)},
    28: {"SMB1": (928,1184,16,16), "SMB3": (1248,192,16,16), "SMW": (768,1312,16,16), "NSMBU": (256,1264,16,16), "SM3DW": (1296,352,16,16)},
    30: {"SMB1": (1168,584,16,32), "SMB3": (1200,952,16,32), "SMW": (208,1200,16,32), "NSMBU": (256,1168,16,32), "SM3DW": (1216,480,16,32)},
    32: {"SMB1": (560,288,64,64), "SMB3": (64,544,64,64), "SMW": (752,512,64,64), "NSMBU": (64,608,64,64), "SM3DW": (688,576,64,64)},
    33: {"SMB1": (976,1184,16,16), "SMB3": (1248,240,16,16), "SMW": (816,1312,16,16), "NSMBU": (304,1264,16,16), "SM3DW": (1296,400,16,16)},
    34: {"SMB1": (1008,1184,16,16), "SMB3": (1248,272,16,16), "SMW": (848,1312,16,16), "NSMBU": (336,1264,16,16), "SM3DW": (1296,432,16,16)},
    35: {"SMB1": (1072,1184,16,16), "SMB3": (1248,336,16,16), "SMW": (912,1312,16,16), "NSMBU": (400,1264,16,16), "SM3DW": (1296,496,16,16)},
    36: {"SMB1": (1088,1184,16,16), "SMB3": (1248,352,16,16), "SMW": (928,1312,16,16), "NSMBU": (416,1264,16,16), "SM3DW": (1296,512,16,16)},
    39: {"SMB1": (624,576,32,32), "SMB3": (1008,32,32,32), "SMW": (1136,96,32,32), "NSMBU": (1040,800,32,32), "SM3DW": (976,256,32,64)},
    41: {"SMB1": (736,1200,16,16), "SMB3": (1248,512,16,16), "SMW": (1088,1312,16,16), "NSMBU": (576,1264,16,16), "SM3DW": (1296,672,16,16)},
    42: {"SMB1": (912,896,32,32), "SMB3": (1008,96,32,32), "SMW": (1136,160,32,32), "NSMBU": (1040,864,32,32), "SM3DW": (608,1072,32,32)},
    45: {"SMB1": (1184,616,16,32), "SMB3": (1184,1000,16,32), "SMW": (224,1200,16,32), "NSMBU": (288,1168,16,32), "SM3DW": (1216,512,16,32)},
    46: {"SMB1": (1200,648,16,32), "SMB3": (1168,1032,16,32), "SMW": (256,1200,16,32), "NSMBU": (320,1168,16,32), "SM3DW": (960,416,16,64)},
    47: {"SMB1": (1008,1200,16,16), "SMB3": (1248,624,16,16), "SMW": (1200,1312,16,16), "NSMBU": (688,1264,16,16), "SM3DW": (1296,816,16,16)},
    48: {"SMB1": (1168,1200,16,16), "SMB3": (1248,784,16,16), "SMW": (1328,32,16,16), "NSMBU": (848,1264,16,16), "SM3DW": (1296,976,16,16)},
    52: {"SMB1": (1184,680,16,32), "SMB3": (1200,1032,16,32), "SMW": (288,1200,16,32), "NSMBU": (352,1168,16,32), "SM3DW": (1216,736,16,32)},
    54: {"SMB1": (1232,32,16,16), "SMB3": (1248,880,16,16), "SMW": (1328,128,16,16), "NSMBU": (944,1264,16,16), "SM3DW": (1296,1072,16,16)},
    55: {"SMB1": (1168,712,16,32), "SMB3": (1184,1064,16,32), "SMW": (320,1200,16,32), "NSMBU": (384,1168,16,32), "SM3DW": (1216,800,16,32)},
    56: {"SMB1": (1232,96,16,16), "SMB3": (1248,944,16,16), "SMW": (1328,192,16,16), "NSMBU": (1008,1264,16,16), "SM3DW": (1296,1136,16,16)},
    58: {"SMB1": (1168,744,16,32), "SMB3": (1184,1096,16,32), "SMW": (368,1200,16,32), "NSMBU": (432,1168,16,32), "SM3DW": (1216,896,16,32)},
    60: {"SMB1": (1232,144,16,16), "SMB3": (1248,992,16,16), "SMW": (1328,240,16,16), "NSMBU": (1056,1264,16,16), "SM3DW": (1296,1184,16,16)},
    61: {"SMB1": (1232,160,16,16), "SMB3": (1248,1008,16,16), "SMW": (1328,256,16,16), "NSMBU": (1072,1264,16,16), "SM3DW": (1296,1200,16,16)},
    62: {"SMB1": (96,944,32,32), "SMB3": (192,800,48,48), "SMW": (192,896,48,48), "NSMBU": (1040,928,32,32), "SM3DW": (672,1072,32,32)},
    67: {"SMB1": (416,944,32,32), "SMB3": (1008,448,32,32), "SMW": (1136,544,32,32), "NSMBU": (224,1040,32,32), "SM3DW": (992,1072,32,32)},
    68: {"SMB1": (864,512,48,48), "SMB3": (240,800,48,48), "SMW": (240,896,48,48), "NSMBU": (288,848,48,48), "SM3DW": (912,240,48,48)},
    76: {"SMB1": (1232,272,16,16), "SMB3": (1248,1120,16,16), "SMW": (1328,352,16,16), "NSMBU": (1184,1264,16,16), "SM3DW": (160,1296,16,16)},
    77: {"SMB1": (672,944,32,32), "SMB3": (1008,704,32,32), "SMW": (1136,800,32,32), "NSMBU": (480,1040,32,32), "SM3DW": (1104,160,32,32)},
    81: {"SMB1": (1232,320,16,16), "SMB3": (1248,1168,16,16), "SMW": (1328,400,16,16), "NSMBU": (1232,1264,16,16), "SM3DW": (208,1296,16,16)},
    83: {"SMB1": (320,480,64,64), "SMB3": (624,64,64,64), "SMW": (384,736,64,64), "NSMBU": (688,0,64,64), "SM3DW": (512,672,64,64)},
    90: {"SMB1": (928,944,32,32), "SMB3": (1008,896,32,32), "SMW": (1136,992,32,32), "NSMBU": (672,1040,32,32), "SM3DW": (1104,352,32,32)},
    98: {"SMB1": (320,976,32,32), "SMB3": (288,1008,32,32), "SMW": (256,1136,32,32), "NSMBU": (1072,0,32,32), "SM3DW": (1104,608,32,32)},
    99: {"SMB1": (1232,688,16,16), "SMB3": (160,1248,16,16), "SMW": (1328,640,16,16), "NSMBU": (1280,192,16,16), "SM3DW": (448,1296,16,16)},
    102: {"SMB1": (1200,488,16,16), "SMB3": (704,1232,16,16), "SMW": (1312,1280,16,16), "NSMBU": (1264,784,16,16), "SM3DW": (896,1280,16,16)},
    104: {"SMB1": (976,432,32,32), "SMB3": (352,976,32,32), "SMW": (576,1104,32,32), "NSMBU": (1040,160,32,32), "SM3DW": (1072,768,32,32)},
    105: {"SMB1": (816,0,48,80), "SMB3": (816,160,48,80), "SMW": (864,80,48,80), "NSMBU": (816,400,48,80), "SM3DW": (816,640,48,80)},
}

# ---------------------------------------------------------------------------
# Autotiling tables for multi-tile *tile-type* structures (ground, pipe,
# mushroom platform, semisolid, bridge). These objects have NO spritesheet
# sprite -- toost draws them from the gamestyle tilesheet, picking a different
# 16px cell per position so the structure reads as one shape (grass top vs.
# interior, pipe rim vs. body, mushroom cap vs. stem) instead of one repeated
# cell. Ported verbatim from toost's LevelDrawer.cpp (Setup / DrawGrdCode /
# DrawTile) so the ascii browser renders them the same way toost.exe does.
#
# Ground uses an 8-neighbour bitmask -> GS[] -> tilesheet cell. The 256-entry
# GS[] table is copied verbatim from LevelDrawer::Setup; each byte packs the
# cell as (X = byte >> 4, Y = byte & 0x0F). Parsed (not hand-indexed) to avoid
# transcription drift; if the count isn't 256 the ground autotiler is disabled
# and '#' falls back to the single interior cell.
_MM2_GS_RAW = """
0x0D, 0x4D, 0x1D, 0xAD, 0x3D, 0x9D, 0x2D, 0xCD, 0x6D, 0x5D, 0x8D, 0xED, 0x7D, 0xDD, 0xBD,
0xFD, 0x0D, 0x4D, 0x1D, 0x2F, 0x3D, 0x9D, 0x2D, 0x4E, 0x6D, 0x5D, 0x8D, 0x0E, 0x7D, 0xDD, 0xBD, 0x8E, 0x0D,
0x4D, 0x1D, 0xAD, 0x3D, 0x4F, 0x2D, 0x5E, 0x6D, 0x5D, 0x8D, 0xED, 0x7D, 0x1E, 0xBD, 0x9E, 0x0D, 0x4D, 0x1D,
0x2F, 0x3D, 0x4F, 0x2D, 0x3F, 0x6D, 0x5D, 0x8D, 0x0E, 0x7D, 0x1E, 0xBD, 0xCE, 0x0D, 0x4D, 0x1D, 0xAD, 0x3D,
0x9D, 0x2D, 0xCD, 0x6D, 0x5D, 0x8F, 0x2E, 0x7D, 0xDD, 0x6E, 0xAE, 0x0D, 0x4D, 0x1D, 0x2F, 0x3D, 0x9D, 0x2D,
0x4E, 0x6D, 0x5D, 0x8F, 0x5F, 0x7D, 0xDD, 0x6E, 0xEE, 0x0D, 0x4D, 0x1D, 0xAD, 0x3D, 0x4F, 0x2D, 0x5E, 0x6D,
0x5D, 0x8F, 0x2E, 0xAF, 0x1E, 0x6E, 0x1F, 0x0D, 0x4D, 0x1D, 0x2F, 0x3D, 0x4F, 0x2D, 0x3F, 0x6D, 0x5D, 0x8F,
0x5F, 0x7D, 0x1E, 0x6E, 0xBF, 0x0D, 0x4D, 0x1D, 0xAD, 0x3D, 0x9D, 0x2D, 0xCD, 0x6D, 0x5D, 0x8D, 0xED, 0xAF,
0x3E, 0x7E, 0xBE, 0x0D, 0x4D, 0x1D, 0x2F, 0x3D, 0x9D, 0x2D, 0x4E, 0x6D, 0x5D, 0x8D, 0x0E, 0x7D, 0x3E, 0x7E,
0x0F, 0x0D, 0x4D, 0x1D, 0xAD, 0x3D, 0x4F, 0x2D, 0x5E, 0x6D, 0x5D, 0x8D, 0xED, 0xAF, 0x7F, 0x7E, 0xFE, 0x0D,
0x4D, 0x1D, 0x2F, 0x3D, 0x4F, 0x2D, 0x3F, 0x6D, 0x5D, 0x8D, 0x0E, 0xAF, 0x7F, 0x7E, 0xCF, 0x0D, 0x4D, 0x1D,
0xAD, 0x3D, 0x9D, 0x2D, 0xCD, 0x6D, 0x5D, 0x8F, 0x2E, 0xAF, 0x3E, 0x9F, 0xDE, 0x0D, 0x4D, 0x1D, 0x2F, 0x3D,
0x9D, 0x2D, 0x4E, 0x6D, 0x5D, 0x8F, 0x5F, 0xAF, 0x3E, 0x9F, 0xDF, 0x0D, 0x4D, 0x1D, 0xAD, 0x3D, 0x4F, 0x2D,
0x5E, 0x6D, 0x5D, 0x8F, 0x2E, 0xAF, 0x7F, 0x9F, 0xEF, 0x0D, 0x4D, 0x1D, 0x2F, 0x3D, 0x4F, 0x2D, 0x3F, 0x6D,
0x5D, 0x8F, 0x5F, 0xAF, 0x7F, 0x9F, 0x6F
"""
_MM2_GROUND_GS = [int(t, 16) for t in re.findall(r"0x[0-9A-Fa-f]{2}", _MM2_GS_RAW)]

# Green pipe cells (PipeLoc[0] in LevelDrawer::Setup), indexed exactly as toost:
# 0=top mouth, 1=bottom mouth, 2=left mouth, 3=right mouth, 4=horizontal body
# (1 wide x 2 tall), 5=vertical body (2 wide x 1 tall). Reconstructed pipes are
# always green (flag PP bits = 0; see mm2pipeline_data.ascii DEFAULT_FLAG).
_MM2_PIPE_LOC = [(14, 0), (14, 2), (11, 0), (13, 0), (12, 0), (14, 1)]

# Multi-tile sprite stamping policy (Pass 1 of _render_mm2_samples), keyed by the
# tileset glyph. Determines what a connected block of the glyph means:
#
#   _MM2_BOSS_GLYPHS  -- ONE large entity that may be painted at different sizes
#       (Bowser at 2x2 or 4x4, a Banzai Bill, ...). The whole connected block is
#       rendered as a SINGLE sprite scaled to fill it, however big the block is,
#       because two adjacent same-glyph cells are still one boss, not two.
#   _MM2_TILED_GLYPHS -- a FIXED-footprint sprite that tiles: a block larger than
#       the footprint is several separate copies (a 4x2 patch of 't' is TWO 2x2
#       Thwomps), so the sprite is stamped footprint-by-footprint across the block.
#
# Glyphs in neither set keep the default "anchor" behaviour: stamp once when the
# block fits the footprint, else fall through to the per-cell fitter -- so a row
# of 1x2 Koopas stays a row of fitted koopas instead of stretched/duplicated ones.
_MM2_BOSS_GLYPHS = {"X", "x", "!", "Z", "A", ";", "z"}  # Bowser, Bowser Jr, Boom Boom, Banzai Bill, Angry Sun, Clown Car, Goomba's Shoe / Yoshi's Egg
_MM2_TILED_GLYPHS = {"t", "%", "j", "0", "f"}       # Thwomp, Saw, Swinging Claw, Skewer, Checkpoint


def _load_mm2_spritesheet():
    """Load img/spritesheet.png once into the module-level cache.

    The sprite *coordinates* are baked into _MM2_SPRITE_DATA (no LevelData.hpp
    needed at runtime); this only opens the PNG they index into. Searched in the
    same spots the rest of the toolchain uses; if it's missing _mm2_spritesheet
    stays None and mm2_tiles() falls back to colour tiles.
    """
    global _mm2_spritesheet
    if _mm2_spritesheet is not None:
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sheet_path = next((p for p in (
        os.path.join(script_dir, "img", "spritesheet.png"),
        os.path.join(script_dir, "toost_stuff", "img", "spritesheet.png"),
    ) if os.path.exists(p)), None)
    if not sheet_path:
        print("mm2_tiles: img/spritesheet.png not found; falling back to colour tiles.")
        return

    _mm2_spritesheet = Image.open(sheet_path).convert("RGBA")


def _mm2_sprite_coords(obj_id, gamestyle):
    """Return the (x, y, w, h) spritesheet rect for obj_id, or None.

    Tries the requested gamestyle first, then the fallback order, since some
    objects only have art under certain styles. Pure data lookup against
    _MM2_SPRITE_DATA -- no spritesheet image required.
    """
    by_style = _MM2_SPRITE_DATA.get(obj_id)
    if not by_style:
        return None
    for style in (gamestyle,) + tuple(s for s in _MM2_STYLE_FALLBACK if s != gamestyle):
        coords = by_style.get(style)
        if coords is not None:
            return coords
    return None


def _mm2_fit(crop, dim, sky):
    """Fit an RGBA sprite inside one dim x dim cell, then return an RGB tile.

    The sprite is scaled to fit the cell preserving aspect ratio -- so a 64x64
    Banzai Bill shrinks to fill the cell and a 16x32 Koopa becomes 8x16 -- with NO
    distortion and NO overflow into neighbouring cells (each cell stays faithful
    to its single tile id). It is anchored bottom-centre on the sky colour so
    ground objects sit on the cell floor; transparent areas show the sky.
    """
    w, h = crop.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (dim, dim), sky)
    scale = min(dim / w, dim / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    if (nw, nh) != (w, h):
        crop = crop.resize((nw, nh), Image.NEAREST)
    tile = Image.new("RGBA", (dim, dim), sky + (255,))
    tile.paste(crop, ((dim - nw) // 2, dim - nh), crop)  # bottom-centre, alpha mask
    return tile.convert("RGB")


def _mm2_sprite_for_object(obj_id, gamestyle, dim, sky):
    """Return a dim x dim RGB tile for an MM2 object id, or None if unavailable.

    Tries the requested gamestyle first, then the fallback order, since some
    objects only have sprites under certain styles. The cropped sprite (whatever
    its native footprint -- a 16x32 Koopa, a 64x64 Banzai Bill, ...) is fit inside
    a single cell by _mm2_fit: scaled to fit, undistorted, never overflowing into
    the neighbouring cells that hold their own tiles.
    """
    if _mm2_spritesheet is None:
        return None

    coords = _mm2_sprite_coords(obj_id, gamestyle)
    if coords is None:
        return None

    x, y, w, h = coords
    if w <= 0 or h <= 0:
        return None
    return _mm2_fit(_mm2_spritesheet.crop((x, y, x + w, y + h)), dim, sky)


def _mm2_get_tilesheet(gamestyle, theme=0):
    """Return a cached RGBA tilesheet (img/tile/<raw>-<theme>.png), or None.

    Block/coin/ground tiles are drawn from this per-gamestyle/theme sheet rather
    than the spritesheet (see _MM2_TILESHEET_CELL).
    """
    key = (gamestyle, theme)
    if key in _mm2_tilesheet_cache:
        return _mm2_tilesheet_cache[key]
    sheet = None
    gs_raw = _MM2_GAMESTYLE_RAW.get(gamestyle)
    if gs_raw is not None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        name = f"{gs_raw}-{theme}.png"
        path = next((p for p in (
            os.path.join(script_dir, "img", "tile", name),
            os.path.join(script_dir, "toost_stuff", "img", "tile", name),
        ) if os.path.exists(p)), None)
        if path:
            sheet = Image.open(path).convert("RGBA")
    _mm2_tilesheet_cache[key] = sheet
    return sheet


def _mm2_tile_from_sheet(cell, gamestyle, dim, sky, theme=0):
    """Return a dim x dim RGB tile cropped from the gamestyle tilesheet, or None.

    Tilesheet cells are a single tile, so this fills the cell exactly; the cell's
    own transparency (e.g. a coin's rounded corners) shows the sky backdrop.
    """
    sheet = _mm2_get_tilesheet(gamestyle, theme)
    if sheet is None:
        return None
    tw = sheet.width // 16
    cx, cy = cell
    return _mm2_fit(sheet.crop((cx * tw, cy * tw, cx * tw + tw, cy * tw + tw)), dim, sky)


def _mm2_glyph_objids():
    """Return (chars, char_to_objid) for the MM2 tileset.

    `chars` is the glyph order extract_tileset() uses (sorted glyphs + the '_'
    padding tile), so index i is model tile id i. `char_to_objid` maps each glyph
    to its MM2 object id (or None for sky/ground/unresolved), via OBJ_META (first
    name listed for a glyph wins, as in mm2pipeline_data.tiles.build_char_to_name) ->
    NAME_TO_ID. Shared by mm2_tiles() and the block-stamping renderer so they
    never drift.
    """
    tileset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), common_settings.MM2_TILESET)
    with open(tileset_path, "r", encoding="utf-8") as f:
        tiles = json.load(f)["tiles"]
    chars = sorted(tiles.keys())
    if "_" not in chars:
        chars = chars + ["_"]

    from mm2pipeline_data.tiles import OBJ_META, NAME_TO_ID
    char_to_name = {}
    for name, (ch, _color, _cat) in OBJ_META.items():
        if name == "_unknown":
            continue
        char_to_name.setdefault(ch, name)

    char_to_objid = {}
    for ch in chars:
        name = char_to_name.get(ch)
        char_to_objid[ch] = NAME_TO_ID.get(name) if name else None
    return chars, char_to_objid


def _mm2_glyph_colors():
    """Return {glyph: (r, g, b)} from OBJ_META's representative colours.

    Used as the graceful fallback when an object's real sprite/tile can't be
    loaded (e.g. assets missing on a machine without the toost data): a square
    colour-coded to the object reads far better than a field of identical grey
    squares. First OBJ_META name per glyph wins, matching _mm2_glyph_objids.
    """
    from mm2pipeline_data.tiles import OBJ_META
    colors = {}
    for name, (ch, color_hex, _cat) in OBJ_META.items():
        if name == "_unknown" or ch in colors:
            continue
        hx = color_hex.lstrip("#")
        try:
            colors[ch] = (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))
        except (ValueError, IndexError):
            continue
    return colors


def _mm2_native_sprite(obj_id, gamestyle):
    """Return (rgba_sprite, (tw, th)) for an object at its native footprint.

    Unlike _mm2_sprite_for_object (which fits the sprite into one cell), this keeps
    the sprite at native resolution (16 px / tile) and reports its nominal tile
    footprint -- a Thwomp is (2, 2), a Skewer (4, 4), a Swinging Claw (3, 5). The
    renderer rescales it to whatever the glyph block actually occupies in the grid
    (which can differ -- claws are painted 3x4 -- and may have overwritten cells).
    Returns None if the object has no spritesheet sprite.
    """
    if _mm2_spritesheet is None or obj_id is None:
        return None
    coords = _mm2_sprite_coords(obj_id, gamestyle)
    if coords is None:
        return None
    x, y, w, h = coords
    if w <= 0 or h <= 0:
        return None
    tw, th = max(1, round(w / 16)), max(1, round(h / 16))
    return _mm2_spritesheet.crop((x, y, x + w, y + h)), (tw, th)


def mm2_tiles(gamestyle=None):
    """Map mm2_tileset_we.json glyphs to MM2 sprites (img/spritesheet.png + tiles).

    Returns a list of dim x dim RGB tile images indexed exactly like
    extract_tileset() orders the tileset (sorted glyphs, then the '_' padding
    tile), so element i is the sprite for model tile id i. Enemies/items come from
    the spritesheet, blocks/coins/pipes/ground from the gamestyle tilesheet; every
    sprite is fit inside its own cell (undistorted, no overflow -- see _mm2_fit).
    Air/padding render as sky blue and anything unresolved as grey.
    """
    if gamestyle is None:
        gamestyle = common_settings.MM2_GAMESTYLE
    dim = common_settings.MM2_TILE_PIXEL_DIM
    sky = common_settings.MM2_SKY_COLOR

    _load_mm2_spritesheet()
    chars, char_to_objid = _mm2_glyph_objids()
    glyph_colors = _mm2_glyph_colors()

    sky_tile = Image.new("RGB", (dim, dim), sky)
    grey_tile = Image.new("RGB", (dim, dim), (128, 128, 128))
    # Ground '#': the fully-surrounded interior tile from the gamestyle tilesheet;
    # solid theme-brown if the tilesheet is missing.
    ground_tile = (_mm2_tile_from_sheet(_MM2_GROUND_CELL, gamestyle, dim, sky)
                   or Image.new("RGB", (dim, dim), (0x8B, 0x69, 0x14)))

    # When the real sprite/tile can't be loaded, show the object's OBJ_META colour
    # (a recognisable colour-coded square) rather than an anonymous grey one.
    color_cache = {}
    def fallback_tile(ch):
        rgb = glyph_colors.get(ch)
        if rgb is None:
            return grey_tile
        if ch not in color_cache:
            color_cache[ch] = Image.new("RGB", (dim, dim), rgb)
        return color_cache[ch]

    tile_images = []
    for ch in chars:
        if ch == " " or ch == "_":
            tile_images.append(sky_tile)
            continue
        if ch == "#":
            tile_images.append(ground_tile)
            continue
        obj_id = char_to_objid.get(ch)
        tile = None
        if obj_id is not None:
            # Enemies/items live in the spritesheet; blocks/coins/pipes/etc. are
            # tilesheet cells (no spritesheet entry) -- try sprite first, then tile.
            tile = _mm2_sprite_for_object(obj_id, gamestyle, dim, sky)
            if tile is None and obj_id in _MM2_TILESHEET_CELL:
                tile = _mm2_tile_from_sheet(_MM2_TILESHEET_CELL[obj_id], gamestyle, dim, sky)
        tile_images.append(tile if tile is not None else fallback_tile(ch))

    return tile_images


def _mm2_multitile_sprites(gamestyle):
    """Return {tile_id: (rgba_sprite, (tw, th))} for objects spanning >1 cell.

    Only multi-tile objects (Thwomp 2x2, Skewer 4x4, Koopa 1x2, ...) are included;
    1x1 objects are handled by the per-cell mm2_tiles() path. The sprite is native
    resolution; (tw, th) is its nominal footprint. Keyed by model tile id.
    """
    chars, char_to_objid = _mm2_glyph_objids()
    out = {}
    for tid, ch in enumerate(chars):
        if ch in (" ", "_", "#"):
            continue
        res = _mm2_native_sprite(char_to_objid.get(ch), gamestyle)
        if res is None:
            continue
        sprite, (tw, th) = res
        if tw > 1 or th > 1:
            out[tid] = (sprite, (tw, th))
    return out


def _mm2_components(grid, tid, consumed, h, w):
    """4-connected components of cells equal to `tid` and not yet consumed."""
    seen = [[False] * w for _ in range(h)]
    comps = []
    for sr in range(h):
        for sc in range(w):
            if grid[sr][sc] != tid or seen[sr][sc] or consumed[sr][sc]:
                continue
            stack = [(sr, sc)]
            seen[sr][sc] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w and not seen[nr][nc]
                            and not consumed[nr][nc] and grid[nr][nc] == tid):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _mm2_sheet_region(cell, cw, ch, gamestyle, theme=0):
    """Crop a cw x ch (in tiles) region anchored at tilesheet cell (cx, cy).

    Returns the RGBA crop at native resolution (so it can be alpha-composited
    over the sky-blue canvas), or None if the tilesheet is unavailable. Used for
    autotiled structures whose pieces span more than one tilesheet cell (the
    2-wide pipe rim/body, the 1x2 bridge plank, ...).
    """
    sheet = _mm2_get_tilesheet(gamestyle, theme)
    if sheet is None:
        return None
    tw = sheet.width // 16
    cx, cy = cell
    return sheet.crop((cx * tw, cy * tw, (cx + cw) * tw, (cy + ch) * tw))


def _mm2_paste_region(canvas, region, c, r, cw, ch, ts):
    """Paste an RGBA tilesheet region covering cw x ch grid cells at (c, r).

    The region is nearest-scaled to the target footprint and alpha-composited,
    so transparent pixels keep the canvas' sky colour. No-op if region is None.
    """
    if region is None:
        return
    target = (cw * ts, ch * ts)
    if region.size != target:
        region = region.resize(target, Image.NEAREST)
    canvas.paste(region, (c * ts, r * ts), region)


def _mm2_autotile(grid, canvas, consumed, chars, gamestyle, ts):
    """Render multi-tile tile-structures with edge-aware tiles, marking consumed.

    This is the rendering-side analogue of mm2pipeline_data.ascii.coalesce(): instead
    of one flat cell per glyph, it groups the glyph's cells (8-neighbour bitmask
    for ground, 4-connected components for the rest) and stamps the correct
    *edge* piece for each position, so a ground field gets grass tops and inner
    corners, a 2-wide pipe gets a rim + body, a mushroom platform gets a cap +
    centred stem, a semisolid gets its surface/edge tiles, and a bridge gets its
    end caps. Cells it draws are marked consumed so the per-cell fallback (pass 2
    in _render_mm2_samples) skips them. If the gamestyle tilesheet is missing it
    bails and every glyph keeps the old single-cell rendering.
    """
    if not grid or not grid[0]:
        return
    if _mm2_get_tilesheet(gamestyle) is None:
        return
    h, w = len(grid), len(grid[0])
    tid_of = {ch: i for i, ch in enumerate(chars)}
    g_ground = tid_of.get("#")
    g_pipe = tid_of.get("|")
    g_mush = tid_of.get("T")
    g_semi = tid_of.get("k")
    g_bridge = tid_of.get("=")

    # --- Ground '#': 8-neighbour bitmask -> GS[] tilesheet cell. ---
    if g_ground is not None and len(_MM2_GROUND_GS) == 256:
        def is_g(r, c):
            return 0 <= r < h and 0 <= c < w and grid[r][c] == g_ground
        for r in range(h):
            for c in range(w):
                if grid[r][c] != g_ground or consumed[r][c]:
                    continue
                # Bit order matches toost's GetGrdCode (note j increases UP there,
                # so toost's "up" == our row-1): corners then orthogonals.
                code = ((int(is_g(r - 1, c - 1)) << 7) | (int(is_g(r - 1, c + 1)) << 6)
                        | (int(is_g(r + 1, c - 1)) << 5) | (int(is_g(r + 1, c + 1)) << 4)
                        | (int(is_g(r - 1, c)) << 3) | (int(is_g(r, c - 1)) << 2)
                        | (int(is_g(r, c + 1)) << 1) | int(is_g(r + 1, c)))
                v = _MM2_GROUND_GS[code]
                region = _mm2_sheet_region((v >> 4, v & 0x0F), 1, 1, gamestyle)
                if region is not None:
                    _mm2_paste_region(canvas, region, c, r, 1, 1, ts)
                    consumed[r][c] = True

    # --- Pipe '|': directional rim + body (LevelDrawer case 9, green PP=0). ---
    if g_pipe is not None:
        for cells in _mm2_components(grid, g_pipe, consumed, h, w):
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            bw, bh = c1 - c0 + 1, r1 - r0 + 1
            if bw == 2 and bh >= 2:           # vertical column -> mouth up
                rim = _mm2_sheet_region(_MM2_PIPE_LOC[0], 2, 1, gamestyle)
                body = _mm2_sheet_region(_MM2_PIPE_LOC[5], 2, 1, gamestyle)
                for r in range(r0, r1 + 1):
                    _mm2_paste_region(canvas, rim if r == r0 else body, c0, r, 2, 1, ts)
                    consumed[r][c0] = consumed[r][c0 + 1] = True
            elif bh == 2 and bw >= 2:         # horizontal run -> mouth right
                rmouth = _mm2_sheet_region(_MM2_PIPE_LOC[3], 1, 2, gamestyle)
                body = _mm2_sheet_region(_MM2_PIPE_LOC[4], 1, 2, gamestyle)
                for c in range(c0, c1 + 1):
                    _mm2_paste_region(canvas, rmouth if c == c1 else body, c, r0, 1, 2, ts)
                    consumed[r0][c] = consumed[r0 + 1][c] = True
            # malformed pipe block: leave to the per-cell fallback (body cell).

    # --- Mushroom Platform 'T': wide cap (top row) + centred stem below. ---
    if g_mush is not None:
        for cells in _mm2_components(grid, g_mush, consumed, h, w):
            r0 = min(p[0] for p in cells)
            cap_cols = sorted(c for r, c in cells if r == r0)
            for c in cap_cols:
                if c == cap_cols[0]:
                    cell = (3, 2)             # left cap (also the single-cell case)
                elif c == cap_cols[-1]:
                    cell = (5, 2)             # right cap
                else:
                    cell = (4, 2)             # middle cap
                _mm2_paste_region(canvas, _mm2_sheet_region(cell, 1, 1, gamestyle), c, r0, 1, 1, ts)
                consumed[r0][c] = True
            for r, c in cells:
                if r == r0:
                    continue
                cell = (6, 3) if r == r0 + 1 else (6, 4)   # stem top vs. body
                _mm2_paste_region(canvas, _mm2_sheet_region(cell, 1, 1, gamestyle), c, r, 1, 1, ts)
                consumed[r][c] = True

    # --- Semisolid 'k': surface/edge tiles over the whole block (case 16). ---
    if g_semi is not None:
        for cells in _mm2_components(grid, g_semi, consumed, h, w):
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            bh = r1 - r0 + 1
            for r, c in cells:
                yy = r - r0
                if yy == 0:
                    oy = 3                    # top surface
                elif yy == 1:
                    oy = 4
                elif yy == bh - 1:
                    oy = 6                    # bottom edge
                else:
                    oy = 5                    # interior fill
                ox = 0 if c == c0 else (2 if c == c1 else 1)
                _mm2_paste_region(canvas, _mm2_sheet_region((7 + ox, oy), 1, 1, gamestyle), c, r, 1, 1, ts)
                consumed[r][c] = True

    # --- Bridge '=': single walkable row, end caps by horizontal adjacency. ---
    if g_bridge is not None:
        for cells in _mm2_components(grid, g_bridge, consumed, h, w):
            cellset = set(cells)
            for r, c in cells:
                if (r, c - 1) not in cellset:
                    cell = (0, 2)             # left end (also the lone-plank case)
                elif (r, c + 1) not in cellset:
                    cell = (2, 2)             # right end
                else:
                    cell = (1, 2)             # middle plank
                _mm2_paste_region(canvas, _mm2_sheet_region(cell, 1, 1, gamestyle), c, r, 1, 1, ts)
                consumed[r][c] = True


def _render_mm2_samples(sample_indices, output_dir, start_index, prompts, gamestyle=None):
    """Render MM2 scenes, reconstructing multi-tile objects from their glyph blocks.

    The forward converter (mm2pipeline_data.ascii) paints each object as a block of its
    glyph -- a Thwomp is a 2x2 patch of 't', a Skewer a 4x4 patch of '0', a
    Swinging Claw a 3x4 patch of 'j'. Three passes turn those blocks back into art:

      Pass 1 walks each multi-tile sprite's connected blocks and stamps them by
        policy (see the glyph sets): a boss fills its whole block with one scaled
        sprite (so a 2x2 and a 4x4 Bowser both render as one boss), a fixed tiled
        sprite repeats footprint-by-footprint (a 4x2 patch of 't' -> two Thwomps),
        and anything else anchors a single native-size stamp, deferring oversized
        blocks (koopa rows, tall poles) to pass 2. Stamped cells are consumed so
        the object takes priority over whatever it overlaps.
      Pass 1b autotiles the tile-type structures (ground, pipe, mushroom platform,
        semisolid, bridge) into edge-aware shapes -- see _mm2_autotile.
      Pass 2 fills everything still unconsumed per-cell (1x1 objects, blocks, and
        block fragments too small to be a real object).

    Mirrors visualize_samples' contract: with output_dir it saves one PNG per scene
    and returns None; otherwise it returns the first scene's image.
    """
    if gamestyle is None:
        gamestyle = common_settings.MM2_GAMESTYLE
    ts = common_settings.MM2_TILE_PIXEL_DIM
    sky = common_settings.MM2_SKY_COLOR

    _load_mm2_spritesheet()
    cell_tiles = mm2_tiles(gamestyle)         # per-cell RGB fallback, by tile id
    multitile = _mm2_multitile_sprites(gamestyle)
    chars, _ = _mm2_glyph_objids()            # tile id -> glyph
    glyph_of = {i: ch for i, ch in enumerate(chars)}
    n = len(cell_tiles)

    first_image = None
    for idx, sample_index in enumerate(sample_indices):
        h, w = sample_index.shape
        grid = [[int(sample_index[r][c]) % n for c in range(w)] for r in range(h)]
        canvas = Image.new("RGB", (w * ts, h * ts), sky)
        consumed = [[False] * w for _ in range(h)]

        # Pass 1: stamp multi-tile sprites across their connected glyph blocks.
        # Each multi-tile glyph follows one of three policies (see the glyph sets
        # above): a boss fills its whole block with one scaled sprite; a fixed
        # tiled sprite repeats footprint-by-footprint across a larger block; and
        # everything else anchors a single native-size stamp (or defers to pass 2).
        for tid, (sprite, (tw, th)) in multitile.items():
            glyph = glyph_of.get(tid)
            is_boss = glyph in _MM2_BOSS_GLYPHS
            is_tiled = glyph in _MM2_TILED_GLYPHS
            # Fixes enemies that should be 2x1 being squished down into a 1x1
            narrow_single = not is_boss and not is_tiled and tw * th <= 2
            for cells in _mm2_components(grid, tid, consumed, h, w):
                # Skip single-cell specks (a stray glyph shouldn't trigger a big
                # cover-up stamp) and scattered noise (require a solid-ish block).
                if len(cells) < 2 and not narrow_single:
                    continue
                rs = [p[0] for p in cells]
                cs = [p[1] for p in cells]
                r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
                bw, bh = c1 - c0 + 1, r1 - r0 + 1
                if len(cells) < 0.5 * bw * bh:
                    continue

                if is_boss:
                    # One sprite scaled to the ENTIRE block, so a 2x2 and a 4x4
                    # Bowser each render as a single (correspondingly sized) boss
                    # instead of bailing to a grid of 1x1 tiles. Covers the bbox.
                    _mm2_paste_region(canvas, sprite, c0, r0, bw, bh, ts)
                    for y in range(r0, r1 + 1):
                        for x in range(c0, c1 + 1):
                            consumed[y][x] = True
                    continue

                if is_tiled:
                    # Fixed footprint: a block larger than tw x th is several
                    # adjacent copies (two 2x2 Thwomps side by side make a 4x2
                    # block -> two stamps), so tile footprint-sized stamps across
                    # it. A block smaller than the footprint clamps to one stamp.
                    cr = r0
                    while cr <= r1:
                        cc = c0
                        while cc <= c1:
                            sw, sh = min(tw, c1 - cc + 1), min(th, r1 - cr + 1)
                            _mm2_paste_region(canvas, sprite, cc, cr, sw, sh, ts)
                            for y in range(cr, cr + sh):
                                for x in range(cc, cc + sw):
                                    consumed[y][x] = True
                            cc += tw
                        cr += th
                    continue

                # Default "anchor": a block bigger than the footprint in either
                # axis is a row/stack of separate objects (a line of koopas), a
                # tall pole, or merged noise -- stamping would only stretch them,
                # so leave those to pass 2.
                if bw > tw or bh > th:
                    continue
                # Render the sprite at its NATIVE footprint (no distortion), anchored
                # bottom-centre over the block, overflowing up/sideways to cover any
                # cells it overlaps (the object takes priority over the backdrop).
                rc0 = max(0, min(round((c0 + c1) / 2 - (tw - 1) / 2), w - tw))
                rr0 = max(0, min(r1 - th + 1, h - th))
                _mm2_paste_region(canvas, sprite, rc0, rr0, tw, th, ts)
                for y in range(rr0, rr0 + th):
                    for x in range(rc0, rc0 + tw):
                        consumed[y][x] = True

        # Pass 1b: autotile multi-tile *tile-type* structures (ground edges,
        # pipe rim/body, mushroom cap/stem, semisolid surface, bridge caps) so
        # they read as one shape instead of a repeated cell. Runs after the
        # sprite stamps so an overflowing boss sprite still takes priority.
        _mm2_autotile(grid, canvas, consumed, chars, gamestyle, ts)

        # Pass 2: per-cell fitted tiles for everything not already stamped.
        for r in range(h):
            for c in range(w):
                if not consumed[r][c]:
                    canvas.paste(cell_tiles[grid[r][c]], (c * ts, r * ts))

        if prompts:
            sanitized_prompt = prompts[idx].replace(".", "")[:50]
            file_name = f"sample_{idx + start_index} - {sanitized_prompt}.png"
        else:
            file_name = f"sample_{idx + start_index} - unconditional.png"

        if output_dir:
            canvas.save(os.path.join(output_dir, file_name))
        elif first_image is None:
            first_image = canvas

    return first_image