"""Hand drawn scenes that check the deterministic MM2 captions still say the right things.

Run from the repo root:
    python -m Game_MM2.verify_mm2_captions
    python -m Game_MM2.verify_mm2_captions --verbose
"""
import sys

import util.common_settings as common_settings
from Game_MM2.MarioMaker_create_ascii_captions import (
    assign_caption, build_id_to_char, get_block_chars, get_char_names,
    get_loose_chars, get_solid_chars, get_tile_categories)

SKY = " " * 20
FLOOR = "#" * 20

# Each check is (name, scene rows, phrases it should say, phrases it shouldn't say).
# Scenes get padded with sky on top so the last row is always the floor.
CHECKS = [
    # floor
    ("full floor", [FLOOR], ["Full ground floor"], []),
    ("floor with a gap", ["########  ##########"], ["Ground floor with one gap"], []),
    ("no floor", ["    ####            "], [], ["ground floor"]),

    # surface shape
    ("ground rising right", [
        "                ####",
        "           #########",
        "      ##############",
        FLOOR], ["Ground rising to the right"], []),
    ("ground sloping down", [
        "####                ",
        "#########           ",
        "##############      ",
        FLOOR], ["Ground sloping down to the right"], []),
    ("one raised section", ["      ######        "] * 4 + [FLOOR] * 3, ["Stepped ground"], ["Uneven"]),
    ("jagged terrain", ["  ##  ##  ##  ##  ##"] * 4 + [FLOOR] * 3, ["Uneven ground"], ["Stepped"]),
    ("flat floor", [FLOOR] * 2, [], ["Uneven", "Stepped", "rising"]),

    # how much ground there is
    ("ground fills most", [FLOOR] * 13, ["fills most of the scene"], []),
    ("raised plateau", ["          ##########"] * 13, ["Raised flat ground"], []),
    ("cave under ground", [
        FLOOR,
        "###             ####",
        FLOOR], ["Open space beneath the ground"], []),

    # valleys
    ("valley", [
        "##                ##",
        "###              ###",
        "####            ####",
        "#####          #####",
        "######        ######",
        "#######      #######",
        "########    ########",
        FLOOR], ["Valley in the middle."], ["Uneven", "Stepped", "rising", "sloping"]),
    ("small valley", [
        "#        ###########",
        "##      ############",
        "###    #############",
        FLOOR], ["Small valley on the left."], []),
    ("lopsided pit valley", [
        "                   #",
        "                   #",
        "                  ##",
        "                  ##",
        "         #       ###",
        "         ##      ###",
        "         ###    ####",
        "#############  #####"],
        ["Valley on the right with a small gap to the bottom, taller on the right."], ["rising", "sloping"]),
    # Pipe standing in the pit means it's not a valley anymore
    ("pipe in the pit", [
        "##                ##",
        "###              ###",
        "####            ####",
        "#####          #####",
        "######        ######",
        "#######      #######",
        "########    ########",
        "########  ||########"], [], ["alley"]),
    ("wide pit", [
        "##                ##",
        "###              ###",
        "####            ####",
        "#####          #####",
        "######        ######",
        "#######      #######",
        "#######      #######"], [], ["alley"]),
    ("trench", ["####    ############"] * 5 + [FLOOR], [], ["alley"]),
    ("basin", [
        "##                ##",
        "###              ###",
        "####            ####",
        "#####  #############",
        FLOOR], [], ["alley"]),
    ("too shallow", [
        "########    ########",
        "#########  #########",
        FLOOR], [], ["alley"]),

    # structures
    ("floating brick platform", [
        "    BBBBB           ",
        SKY,
        FLOOR], ["block platform"], ["block ledge"]),
    ("brick ledge", [
        "    BBBBB           ",
        "    #####           ",
        FLOOR], ["block ledge"], ["block platform"]),
    ("bricks inside a wall", [
        "    #####           ",
        "    BBBBB           ",
        "    #####           ",
        FLOOR], [], ["block platform", "block ledge"]),
    ("tower", ["   HH               "] * 5 + [FLOOR], ["hard block tower"], ["hard block column"]),
    ("floating column", ["   HH               "] * 5 + [SKY] * 3 + [FLOOR], ["hard block column"], ["hard block tower"]),
    ("brick staircase", [
        "        B           ",
        "       BB           ",
        "      BBB           ",
        "     BBBB           ",
        FLOOR], ["block staircase"], []),
    ("L shape", [
        "     H              ",
        "     H              ",
        "     H              ",
        "     HHHHHH         ",
        FLOOR], [], ["staircase"]),
    ("short staircase", [
        "       H            ",
        "      HH            ",
        "     HHH            ",
        FLOOR], ["hard block staircase"], []),
    ("filled in staircase", [
        "        H           ",
        "       HH           ",
        "      HHH           ",
        "     HHHH           ",
        "     HHHH           ",
        FLOOR], ["hard block staircase"], []),
    ("staircase under a roof", [
        "     ####           ",
        "        B           ",
        "       BB           ",
        "      BBB           ",
        "     BBBB           ",
        FLOOR], [], ["staircase"]),

    # coin and enemy arrangements
    ("row of coins", ["   ccccc            ", FLOOR], ["One line of coins"], []),
    ("two rows of coins", [
        "   ccccc            ",
        SKY,
        "   ccccc            ",
        FLOOR], ["Two lines of coins"], []),
    ("column of coins", ["   c                "] * 4 + [FLOOR], ["One column of coins"], []),
    ("square of coins", ["   ccc              "] * 3 + [FLOOR], ["One cluster of coins"], []),
    ("coin clump", [
        "     cc             ",
        "    cccc            ",
        "     ccc            ",
        "      c             ",
        FLOOR], ["One clump of coins"], []),
    ("scattered coins", [
        "  c   c   c         ",
        "      c     c       ",
        FLOOR], ["Scattered coins"], []),
    ("two coin clumps", [
        "     cc        cc   ",
        "    cccc      cccc  ",
        "     ccc       ccc  ",
        "      c         c   ",
        FLOOR], ["Two clumps of coins"], []),
    ("two coins", ["   cc               ", FLOOR], [], ["line of coins", "Scattered"]),
    ("row of goombas", ["   ggggg            ", FLOOR], ["One line of goombas"], []),

    # big forms
    ("big goomba", ["  gg                "] * 2 + [FLOOR], ["One big goomba"], ["cluster"]),
    ("big goomba and two small", [
        "  gg      g   g     ",
        "  gg                ",
        FLOOR], ["One big goomba", "Two goombas"], []),
    ("two big goombas", ["  gg   gg           "] * 2 + [FLOOR], ["Two big goombas"], []),
    ("big coin", ["  cc                "] * 2 + [FLOOR], ["One big coin"], []),
    ("big goomba touching another", [
        "  ggg               ",
        "  gg                ",
        FLOOR], [], ["big goomba"]),
    ("p switches", ["  SS                "] * 2 + [FLOOR], [], ["big p switch"]),
    ("row of four goombas", ["  gggg              ", FLOOR], ["One line of goombas"], ["big goomba"]),
    ("3x3 goombas", ["  ggg               "] * 3 + [FLOOR], [], ["big goomba"]),

    # plurals
    ("spiny", ["  sss     s         ", FLOOR], ["spinies"], ["spinys"]),
    ("p switch", ["  SSS     S         ", FLOOR], ["p switches"], ["p switchs"]),
    ("rocky wrench", ["  rrr     r         ", FLOOR], ["wrenches"], ["wrenchs"]),
    ("dry bones", ["  999     9         ", FLOOR], ["dry bones"], ["dry boness"]),
    ("bowser jr", ["  x                 ", FLOOR], ["One bowser jr."], [".."]),
    ("no blob", [
        "   ccccc            ",
        "   ggggg            ",
        FLOOR], [], ["blob"]),

    # ceilings
    ("full ceiling", [FLOOR] + [SKY] * 18 + [FLOOR], ["Full ceiling"], []),
    ("half ceiling", ["##########          "] + [SKY] * 18 + [FLOOR], ["Ceiling with"], []),
    ("couple of blocks up top", ["##                  "] + [SKY] * 19, [], ["eiling"]),
    ("all ground", [FLOOR] * 20, [], ["eiling"]),
]


def make_scene(rows, char_to_id):
    grid = [SKY] * (20 - len(rows)) + rows
    return [[char_to_id[char] for char in row] for row in grid]


def main():
    verbose = "--verbose" in sys.argv
    tileset = common_settings.MM2_TILESET

    id_to_char = build_id_to_char(tileset)
    char_to_id = {char: idx for idx, char in id_to_char.items()}
    names = get_char_names(tileset)
    _, _, ground = get_tile_categories(tileset)
    kinds = dict(ground_chars=ground, block_chars=get_block_chars(tileset),
                 solid_chars=get_solid_chars(tileset), loose_chars=get_loose_chars(tileset))

    def caption(rows):
        return assign_caption(make_scene(rows, char_to_id), id_to_char, names, **kinds)

    failed = 0
    for name, rows, should_say, shouldnt_say in CHECKS:
        text = caption(rows)
        missing = [p for p in should_say if p not in text]
        extra = [p for p in shouldnt_say if p in text]
        if missing or extra:
            failed += 1
            print("FAIL", name)
            print("    ", text)
            if missing:
                print("     missing:", missing)
            if extra:
                print("     shouldn't say:", extra)
        elif verbose:
            print("ok  ", name, "-", text)

    # Training data relies on the same scene always getting the same caption
    rows = ["   ccccc            ", "   ggggg            ", FLOOR]
    if caption(rows) != caption(rows):
        failed += 1
        print("FAIL same scene gave two different captions")

    if caption([SKY]).strip():
        failed += 1
        print("FAIL scene of just air got a caption")

    # Ceiling isn't always row 0 so we make sure the highlighted cells are on the row it found
    rows = ["cccccccccccccccccccc", "HHHHHHHHHHHHHHHHHHHH"] + [SKY] * 17 + [FLOOR]
    _, details = assign_caption(make_scene(rows, char_to_id), id_to_char, names, return_details=True, **kinds)
    for phrase, cells in details.items():
        if phrase.startswith(("Full ceiling", "Ceiling with")) and {r for r, _ in cells} != {1}:
            failed += 1
            print("FAIL ceiling cells aren't on row 1")

    if failed:
        print(f"\n{failed} check(s) failed")
        sys.exit(1)
    print(f"\nAll {len(CHECKS)} scenes passed")


if __name__ == "__main__":
    main()
