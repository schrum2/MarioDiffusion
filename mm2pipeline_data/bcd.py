"""Build valid Toost-loadable .bcd course files from decoded MM2 level payloads,
and parse the level.ksy binary structure (gamestyle, object ids) needed to
filter levels during extraction.

Background
----------
The on-disk .bcd course file is exactly 0x5C000 (376,832) bytes:
  [0x000–0x00F]  16-byte header (unknown / padding)
  [0x010–0x5BFD0-ish]  AES-128-CBC encrypted course payload
  [0x5BFD0–0x5BFFF]  48-byte trailer:
      [+0x00..+0x0F]  IV  (16 bytes)
      [+0x10..+0x1F]  seed words s0..s3 (4 × uint32 LE) used to derive the key
      [+0x20..+0x2F]  CMAC placeholder / zeros

The HuggingFace dataset's `level_data` column is  zlib( decrypted_payload )  where
decrypted_payload starts at offset 0x10 and is 0x5C000-0x40 = 0x5BFC0 bytes.

Key derivation and the XORSHIFT128 PRNG mirror simontime/SMM2CourseDecryptor.

Requirements:
    pip install pycryptodome
"""

import struct
import zlib

# ---------------------------------------------------------------------------
# AES-128-CBC  (requires pycryptodome: pip install pycryptodome)
# ---------------------------------------------------------------------------

def aes_cbc_encrypt(key_bytes: bytes, iv_bytes: bytes, plaintext: bytes) -> bytes:
    from Crypto.Cipher import AES
    cipher = AES.new(key_bytes, AES.MODE_CBC, iv_bytes)
    return cipher.encrypt(plaintext)


# ---------------------------------------------------------------------------
# Key table from simontime/SMM2CourseDecryptor  (keys.h → course_key_table)
# ---------------------------------------------------------------------------

COURSE_KEY_TABLE = [
    0x7AB1C9D2, 0xCA750936, 0x3003E59C, 0xF261014B,
    0x2E25160A, 0xED614811, 0xF1AC6240, 0xD59272CD,
    0xF38549BF, 0x6CF5B327, 0xDA4DB82A, 0x820C435A,
    0xC95609BA, 0x19BE08B0, 0x738E2B81, 0xED3C349A,
    0x045275D1, 0xE0A73635, 0x1DEBF4DA, 0x9924B0DE,
    0x6A1FC367, 0x71970467, 0xFC55ABEB, 0x368D7489,
    0x0CC97D1D, 0x17CC441E, 0x3528D152, 0xD0129B53,
    0xE12A69E9, 0x13D1BDB7, 0x32EAA9ED, 0x42F41D1B,
    0xAEA5F51F, 0x42C5D23C, 0x7CC742ED, 0x723BA5F9,
    0xDE5B99E3, 0x2C0055A4, 0xC38807B4, 0x4C099B61,
    0xC4E4568E, 0x8C29C901, 0xE13B34AC, 0xE7C3F212,
    0xB67EF941, 0x08038965, 0x8AFD1E6A, 0x8E5341A3,
    0xA4C61107, 0xFBAF1418, 0x9B05EF64, 0x3C91734E,
    0x82EC6646, 0xFB19F33E, 0x3BDE6FE2, 0x17A84CCA,
    0xCCDF0CE9, 0x50E4135C, 0xFF2658B2, 0x3780F156,
    0x7D8F5D68, 0x517CBED1, 0x1FCDDF0D, 0x77A58C94,
]

MASK32 = 0xFFFFFFFF


def rand_init(s0, s1, s2, s3):
    cond = s0 | s1 | s2 | s3
    if cond:
        return [s0, s1, s2, s3]
    return [1, 0x6C078967, 0x714ACB41, 0x48077044]


def rand_gen(state):
    n = (state[0] ^ ((state[0] << 11) & MASK32)) & MASK32
    state[0] = state[1]
    n = (n ^ (n >> 8) ^ state[3] ^ (state[3] >> 19)) & MASK32
    state[1] = state[2]
    state[2] = state[3]
    state[3] = n
    return n


def gen_key(key_table, state):
    """Produce a 16-byte AES key (4 × uint32 LE) from the PRNG state."""
    out = [0, 0, 0, 0]
    for i in range(4):
        for _ in range(4):
            out[i] = (out[i] << 8) & MASK32
            idx   = rand_gen(state) >> 26          # 6-bit index into table (64 entries)
            shift = (rand_gen(state) >> 27) & 24   # shift ∈ {0, 8, 16, 24}
            out[i] |= (key_table[idx] >> shift) & 0xFF
    return struct.pack("<4I", *out)


# ---------------------------------------------------------------------------
# Build a valid encrypted .bcd from a raw decrypted payload
# ---------------------------------------------------------------------------

COURSE_FILE_SIZE  = 0x5C000   # 376,832 bytes
HEADER_SIZE       = 0x10      # 16-byte header we skip / zero-pad
TRAILER_SIZE      = 0x30      # 48-byte trailer: IV(16) + seed(16) + zeros(16)
PAYLOAD_SIZE      = COURSE_FILE_SIZE - HEADER_SIZE - TRAILER_SIZE  # 0x5BFC0

# Offset of the gamestyle field (s2le) within the decompressed payload,
# per level.ksy: sum of the fixed header fields before it (52 bytes) plus
# the 189-byte unk1 padding = 241.
GAMESTYLE_OFFSET  = 241
GAMESTYLE_SM3DW   = 22323   # level.ksy enum gamestyle: sm3dw


def get_gamestyle_raw(plaintext: bytes) -> int:
    """Read the raw gamestyle enum value from a decoded level payload."""
    return struct.unpack_from("<h", plaintext, GAMESTYLE_OFFSET)[0]


# ---------------------------------------------------------------------------
# Object IDs (level.ksy: obj_id enum) and item-based level skipping
# ---------------------------------------------------------------------------

OBJ_ID = {
    "goomba": 0, "koopa": 1, "piranha_flower": 2, "hammer_bro": 3,
    "block": 4, "question_block": 5, "hard_block": 6, "ground": 7,
    "coin": 8, "pipe": 9, "spring": 10, "lift": 11,
    "thwomp": 12, "bullet_bill_blaster": 13, "mushroom_platform": 14, "bob_omb": 15,
    "semisolid_platform": 16, "bridge": 17, "p_switch": 18, "pow": 19,
    "super_mushroom": 20, "donut_block": 21, "cloud": 22, "note_block": 23,
    "fire_bar": 24, "spiny": 25, "goal_ground": 26, "goal": 27,
    "buzzy_beetle": 28, "hidden_block": 29, "lakitu": 30, "lakitu_cloud": 31,
    "banzai_bill": 32, "one_up": 33, "fire_flower": 34, "super_star": 35,
    "lava_lift": 36, "starting_brick": 37, "starting_arrow": 38, "magikoopa": 39,
    "spike_top": 40, "boo": 41, "clown_car": 42, "spikes": 43,
    "big_mushroom": 44, "shoe_goomba": 45, "dry_bones": 46, "cannon": 47,
    "blooper": 48, "castle_bridge": 49, "jumping_machine": 50, "skipsqueak": 51,
    "wiggler": 52, "fast_conveyor_belt": 53, "burner": 54, "door": 55,
    "cheep_cheep": 56, "muncher": 57, "rocky_wrench": 58, "track": 59,
    "lava_bubble": 60, "chain_chomp": 61, "bowser": 62, "ice_block": 63,
    "vine": 64, "stingby": 65, "arrow": 66, "one_way": 67,
    "saw": 68, "player": 69, "big_coin": 70, "half_collision_platform": 71,
    "koopa_car": 72, "cinobio": 73, "spike_ball": 74, "stone": 75,
    "twister": 76, "boom_boom": 77, "pokey": 78, "p_block": 79,
    "sprint_platform": 80, "smb2_mushroom": 81, "donut": 82, "skewer": 83,
    "snake_block": 84, "track_block": 85, "charvaargh": 86, "slight_slope": 87,
    "steep_slope": 88, "reel_camera": 89, "checkpoint_flag": 90, "seesaw": 91,
    "red_coin": 92, "clear_pipe": 93, "conveyor_belt": 94, "key": 95,
    "ant_trooper": 96, "warp_box": 97, "bowser_jr": 98, "on_off_block": 99,
    "dotted_line_block": 100, "water_marker": 101, "monty_mole": 102, "fish_bone": 103,
    "angry_sun": 104, "swinging_claw": 105, "tree": 106, "piranha_creeper": 107,
    "blinking_block": 108, "sound_effect": 109, "spike_block": 110, "mechakoopa": 111,
    "crate": 112, "mushroom_trampoline": 113, "porkupuffer": 114, "cinobic": 115,
    "super_hammer": 116, "bully": 117, "icicle": 118, "exclamation_block": 119,
    "lemmy": 120, "morton": 121, "larry": 122, "wendy": 123,
    "iggy": 124, "roy": 125, "ludwig": 126, "cannon_box": 127,
    "propeller_box": 128, "goomba_mask": 129, "bullet_bill_mask": 130, "red_pow_box": 131,
    "on_off_trampoline": 132,
}

# Object names that, if present anywhere in a level (overworld or subworld),
# cause that level to be skipped during extraction, the same way 3D World
# levels are skipped. Add more names from OBJ_ID above as needed.
SKIP_ITEM_NAMES = [
    "fast_conveyor_belt",
    "conveyor_belt",
    "track",
    "track_block",
    "snake_block",
    "smb2_mushroom",  # v3.0.0 style power-up: no SMM:WE equivalent, levels often depend on it
]

SKIP_OBJECT_IDS = {OBJ_ID[name] for name in SKIP_ITEM_NAMES}

# Layout of the "map" (overworld/subworld) structure within the decoded
# payload, derived from level.ksy.
OVERWORLD_OFFSET          = 512      # start of the overworld map
MAP_OBJECT_COUNT_OFF      = 28       # s4 object_count, relative to map start
MAP_SNAKE_BLOCK_COUNT_OFF = 36       # s4 snake_block_count (level.ksy)
MAP_TRACK_BLOCK_COUNT_OFF = 52       # s4 track_block_count (level.ksy)
MAP_TRACK_COUNT_OFF       = 64       # s4 track_count (level.ksy)
MAP_OBJECTS_OFF           = 72       # start of the fixed-size objects array
MAP_OBJ_ENTRY_SIZE        = 32       # bytes per `obj` entry
MAP_OBJ_ID_OFF            = 24       # offset of `id` (s2) within an obj entry
MAP_MAX_OBJECTS           = 2600     # fixed array length (level.ksy)
MAP_SIZE                  = 188128   # total size of one map (overworld/subworld)
SUBWORLD_OFFSET           = OVERWORLD_OFFSET + MAP_SIZE

# Tracks, snake blocks, and track blocks are stored in dedicated arrays, NOT
# as regular obj entries — so they have no obj_id in the objects array.
# Map these item names to the count field offset that reliably detects them.
_COUNT_FIELD_NAMES = {
    "track":       MAP_TRACK_COUNT_OFF,
    "snake_block": MAP_SNAKE_BLOCK_COUNT_OFF,
    "track_block": MAP_TRACK_BLOCK_COUNT_OFF,
}
SKIP_COUNT_OFFSETS = {
    off for name, off in _COUNT_FIELD_NAMES.items() if name in SKIP_ITEM_NAMES
}


def _map_object_ids(plaintext: bytes, map_offset: int):
    object_count = struct.unpack_from("<i", plaintext, map_offset + MAP_OBJECT_COUNT_OFF)[0]
    object_count = max(0, min(object_count, MAP_MAX_OBJECTS))
    base = map_offset + MAP_OBJECTS_OFF
    for i in range(object_count):
        yield struct.unpack_from("<h", plaintext, base + i * MAP_OBJ_ENTRY_SIZE + MAP_OBJ_ID_OFF)[0]


def level_contains_skip_object(plaintext: bytes) -> bool:
    for map_offset in (OVERWORLD_OFFSET, SUBWORLD_OFFSET):
        for off in SKIP_COUNT_OFFSETS:
            if struct.unpack_from("<i", plaintext, map_offset + off)[0] > 0:
                return True
        for obj_id in _map_object_ids(plaintext, map_offset):
            if obj_id in SKIP_OBJECT_IDS:
                return True
    return False


def subworld_has_items(plaintext: bytes) -> bool:
    """Return True if the subworld contains any objects or special blocks."""
    if struct.unpack_from("<i", plaintext, SUBWORLD_OFFSET + MAP_OBJECT_COUNT_OFF)[0] > 0:
        return True
    for off in (MAP_SNAKE_BLOCK_COUNT_OFF, MAP_TRACK_BLOCK_COUNT_OFF, MAP_TRACK_COUNT_OFF):
        if struct.unpack_from("<i", plaintext, SUBWORLD_OFFSET + off)[0] > 0:
            return True
    return False


def build_bcd(plaintext: bytes) -> bytes:
    """
    Given the decrypted course payload (0x5BFC0 bytes), return the full
    0x5C000-byte encrypted .bcd file that Toost can load.
    """
    if len(plaintext) != PAYLOAD_SIZE:
        raise ValueError(
            f"Plaintext must be exactly {PAYLOAD_SIZE} bytes, got {len(plaintext)}"
        )

    # Fixed non-zero seed and IV; toost only needs the payload to decrypt.
    s0, s1, s2, s3 = 0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0
    iv_bytes = b"NintendoMaker201"  # arbitrary, consistent

    state = rand_init(s0, s1, s2, s3)
    key_bytes = gen_key(COURSE_KEY_TABLE, state)

    ciphertext = aes_cbc_encrypt(key_bytes, iv_bytes, plaintext)

    # Trailer layout  (0x30 bytes):
    #   [0x00..0x0F]  IV
    #   [0x10..0x13]  s0 LE  (seed used to derive key)
    #   [0x14..0x17]  s1 LE
    #   [0x18..0x1B]  s2 LE
    #   [0x1C..0x1F]  s3 LE
    #   [0x20..0x2F]  zeros (CMAC placeholder)
    trailer = (
        iv_bytes
        + struct.pack("<4I", s0, s1, s2, s3)
        + b"\x00" * 16
    )

    bcd = b"\x00" * HEADER_SIZE + ciphertext + trailer
    assert len(bcd) == COURSE_FILE_SIZE, f"BCD size mismatch: {len(bcd)}"
    return bcd


def decompress_level_data(raw) -> bytes:
    if isinstance(raw, (list, bytearray)):
        raw = bytes(raw)
    try:
        return zlib.decompress(raw)
    except zlib.error:
        pass
    try:
        return zlib.decompress(raw, 16 + zlib.MAX_WBITS)
    except zlib.error as e:
        raise ValueError(f"Cannot decompress: {e}") from e
