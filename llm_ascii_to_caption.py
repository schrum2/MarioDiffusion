"""
llm_ascii_to_caption.py

Generalized captioning pipeline: loads scenes in VGLC-ASCII / integer-tile-id
format for ANY registered game (see util/descriptive_tilesets.py::GAMES) and
captions them with an LLM.

A loop runs over every scene in the dataset, prompting the LLM with the tileset
key and the ASCII (or token) level grid, and assigns each scene a set of
generated captions. Results are checkpointed to a crash-safe JSONL file as they
complete and rebuilt into --output at the end (and after every run, so a
resumed run's --output always reflects everything finished so far).

Game-specific details (tileset paths, readable tile names, and any
game-specific prompt vocabulary/rules) live in util/descriptive_tilesets.py's
GAMES registry, not in this file -- this file is meant to caption any
registered game unchanged.
"""
from pathlib import Path
from collections import Counter
import os
import json
import time
import argparse
import threading
import urllib.request
import urllib.error

import ollama
from tqdm import tqdm

from create_level_json_data import load_levels
from captions.util import extract_tileset
from util.descriptive_tilesets import GAMES


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

# for ollama
OLLAMA_NUM_CTX = 8192          # floor context window; grown per-scene up to OLLAMA_MAX_NUM_CTX
OLLAMA_MAX_NUM_CTX = 16384     # ceiling context window
EXPECTED_CAPTIONS = 5          # default number of captions expected per LLM query; overridable with --num_captions
MAX_CAPTION_RETRIES = 20       # how many times we tolerate responses with the wrong caption count
MAX_REPROMPTS = 10             # how many times we tolerate empty / non-ASCII responses


# Spelled-out forms for the caption count so the prompts can read naturally ("exactly five
# captions") for any --num_captions value; falls back to the digit for counts outside the map.
_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}


def num_word(n: int) -> str:
    """Spell out a small caption count ('five'); fall back to the digit string for larger counts."""
    return _NUMBER_WORDS.get(n, str(n))


# Default model per --llm branch, used when --model isn't supplied. The claude, openai, and
# gemini branches hit APIs (via --api-key-file); ollama runs a local model.
#
# NOT YET IMPLEMENTED, but the backend-dispatch design below (BACKEND_CALLERS) makes adding
# one of these easy later: write a call_<name>(...) function with the same signature as the
# existing call_* functions, add it to BACKEND_CALLERS, add its default model here, and add
# "<name>" to --llm's `choices` in parse_args(). No other code needs to change.
#   "smolvlm":   "HuggingFaceTB/SmolVLM-Instruct",
#   "moondream": "moondream",
#   "llava":     "llava:7b",
DEFAULT_MODELS = {
    "claude": "claude-haiku-4-5",
    "openai": "gpt-5.1",
    "gemini": "gemini-2.5-flash",
    "ollama": "qwen3.5:9b",
}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_system_prompt(num_captions: int, game_name: str,
                        vocab_extra: list[str] = (), rule_extra: list[str] = ()) -> str:
    """
    Build the captioning system prompt, phrased to request exactly {num_captions} captions
    for {game_name}.

    vocab_extra: game-specific vocabulary/tone/naming guidance (e.g. how to refer to enemies
        in this game). Comes from GAMES[game]["prompt_vocab"]; empty for games that haven't
        had this filled in yet.
    rule_extra: extra RULE-like paragraphs unique to this game (e.g. Mario Maker 2's
        multi-tile-object dedup instruction). Comes from GAMES[game]["prompt_rules"]; empty
        for games that don't need any.
    """
    word = num_word(num_captions)
    vocab_block = "\n".join(v for v in vocab_extra if v)
    rules_block = "\n".join(f"- {r}" for r in rule_extra if r)
    # Built outside the f-string: a literal backslash (the \n here) can't appear inside an
    # f-string's {...} expression part on Python < 3.12, which is what caused the SyntaxError.
    vocab_section = f"\n{vocab_block}\n" if vocab_block else ""

    return f"""
You are a {game_name} captioning agent; given an ASCII (or tokenized) grid representation of a
{game_name} level, a tile set key, and deterministic level data, you must generate EXACTLY
{word.upper()} diverse captions that all describe the level accurately.
{vocab_section}
RULES:
- Your captions should each DISTINCTLY vary in tone, length, wordiness, playfulness, specificity, etc.
while remaining accurate. Make the diversity noticeable, including short and long captions, playfully
descriptive captions and monotone, serious captions, and so on. E.g., in some captions, include specific
enemy names while in others refer to them as ground enemies/flying enemies, etc.
- Do not mention specific tile types in your answer that you see in the tile set (raw symbols/tokens),
just describe the level with words.
- Your captions should primarily focus on level structure, and features in the level, typically
with relative locations, although not explicitly required. Mention specific structures/features like
platforms, enemies, corridors, etc.
- Caption the level like you're writing a prompt to generate it; this means specificity and directness is essential.
- All of the levels you're captioning are roughly square shaped; don't call them "wide" or "tall", as they're square.
This vocabulary could be used for certain structures within the level, like a short horizontal corridor/thin vertical shaft.
{rules_block}

ORIENTATION: The first grid row is the TOP of the level and the last row is the BOTTOM; gravity points
down, toward the last row. The player spawn is where the player STARTS, and the player progresses AWAY from it
toward the far end of the level, towards the level exit. Travel toward the top of the grid is ascending (climbing up);
travel toward the bottom is descending (dropping or falling down). If there is no spawn/exit in the level frame, don't
guess between ascending and descending; simply classify the scene as vertical (chamber, segment, shaft, etc.)

FORMATTING:
- Output a single JSON array of exactly {num_captions} strings, and nothing else -- no markdown fences,
no commentary, no keys other than the array itself.
- Do not number the captions or add bullets; each array element is one complete caption.
- You must write exactly {word.upper()} captions; no more, no less.
- Do not include any dashes or semicolons within a caption. The only punctuation you should
use are commas and periods (, and .). Keep commas rare and only within a single
phrase, and do not chain multiple distinct ideas together with commas.
- Encapsulate each distinct idea or feature in its own concentrated phrase ended by a
period, rather than stringing many ideas into one run-on sentence. Your captions should still vary
freely in tone, length, and wordiness, never homogeneous in format or structure.

REMINDERS:
- Make sure your captions are each NOTICEABLY DISTINCT from one another in length, tone, and specificity:
- For length: at least one caption should be long, one should be short, and the rest should
fall in between.
- For specificity: one or two captions should contain specific enemy/power-up names, while others can be vague and state enemy class (ranged, ground).
- You must return {word.upper()} distinct captions, as elements of one JSON array.
- Don't say things like "This level has..." or "The level features...". Just directly
describe the level itself without mentioning "level".
"""


def build_caption_reminder(num_captions: int) -> str:
    """Trailing reminder appended as the final (freshest) part of the prompt."""
    word = num_word(num_captions)
    return (f"Reminder: output a single JSON array of exactly {word} caption strings and nothing else -- "
    "no markdown fences, no commentary, no numbering, no keys other than the array itself. "
    f"Break each caption into short '.'-separated phrases, and make the {word} intentionally diverse in length, specificity, and tone. "
    "For vertical segments, do not guess at whether the segment is ascending or descending, unless it is absolutely clear by the level structure/metadata "
    "that you observe and are provided. For ambiguous vertical scenes, note the structure and verticality but don't assume directionality. "
    "You may still classify structures within a scene as ascending or descending, especially if there are notable structures "
    "present in horizontal (always left-to-right) level scenes. Don't explicitly state that a level is left-to-right; that information is already known."
    )


def build_avoidance_clause(bad_chars: set) -> str:
    """Build a CRITICAL clause banning every non-ASCII character seen so far this run.

    Returns "" when no bad characters have been collected yet, so a clean run leaves the
    base prompt untouched.
    """
    if not bad_chars:
        return ""
    char_list = ", ".join(f"{ch!r} (U+{ord(ch):04X})" for ch in sorted(bad_chars))
    return (
        f"\n\nCRITICAL: The following non-ASCII characters have appeared in earlier outputs during "
        f"this run and are strictly forbidden: {char_list}. NEVER use any of them. Use only plain "
        f"ASCII characters (code points 0-127) -- for example, a hyphen-minus '-' instead of an em "
        f"dash, and straight quotes instead of curly quotes. Output the JSON array with only "
        f"ASCII characters."
    )


def build_count_clause(num_captions: int, retries: int) -> str:
    """Clause re-stressing the exact caption count, added once a scene has miscounted.

    Escalates with the retry number: a firm reminder for the first couple of misses, then a
    blunter demand plus an explicit N-slot JSON skeleton once the model keeps miscounting.
    Returns "" until the scene has miscounted at least once.
    """
    if not retries:
        return ""
    plural = "caption" if num_captions == 1 else "captions"
    string_plural = "string" if num_captions == 1 else "strings"
    base = (
        f"\n\nCRITICAL: An earlier attempt for this level returned the WRONG number of captions. You "
        f"MUST output EXACTLY {num_captions} {plural} -- no more and no fewer -- as a single JSON array "
        f"of {num_captions} {string_plural}. Count your captions before answering and make sure there "
        f"are exactly {num_captions}."
    )
    if retries < 3:
        return base
    slots = ", ".join(f'"<caption {i + 1}>"' for i in range(num_captions))
    return base + (
        f"\n\nThis has now failed {retries} times. STOP and follow this exactly. Output ONLY a JSON "
        f"array with EXACTLY {num_captions} elements, in this shape, replacing each placeholder with a "
        f"real caption while keeping the brackets and commas:\n[{slots}]\n"
        f"There must be exactly {num_captions} comma-separated {string_plural} between the brackets and "
        f"absolutely nothing before '[' or after ']'."
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_UNICODE_NORMALIZE = {
    '\u2018': "'", '\u2019': "'",   # left/right single quotes
    '\u201c': '"', '\u201d': '"',   # left/right double quotes
    '\u2014': '-', '\u2013': '-',   # em dash, en dash
    '\u2026': '...',                # ellipsis
    '\u00e9': 'e', '\u00e8': 'e',   # accented e
    '\u00e0': 'a', '\u00e2': 'a',   # accented a
    '\u00f4': 'o',                  # accented o
    '\u00fc': 'u', '\u00fb': 'u',   # accented u
    '\u00b7': '.', '\u2022': '.',   # middle dot, bullet
}


def normalize_to_ascii(text: str) -> str:
    """Replace common non-ASCII characters with ASCII equivalents."""
    return ''.join(_UNICODE_NORMALIZE.get(ch, ch) for ch in text)


def find_non_ascii_chars(captions: list[str]) -> list[str]:
    """Return a sorted list of all distinct non-ASCII characters across captions."""
    bad = {ch for caption in captions for ch in caption if ord(ch) > 127}
    return sorted(bad)


def parse_captions(raw_response: str) -> list[str]:
    """
    Parse the LLM's JSON array of captions, with graceful fallbacks.

    Every backend is prompted to return a single JSON array of strings (see
    build_system_prompt's FORMATTING section), so this is the one parsing path used for
    all backends. Returns a list of caption strings (possibly empty if parsing fails
    entirely).
    """
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return [normalize_to_ascii(c).strip() for c in parsed if c.strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: try to find the first [...] block in the response.
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return [normalize_to_ascii(c).strip() for c in parsed if c.strip()]
        except json.JSONDecodeError:
            pass

    # Last resort: treat each non-empty line as its own caption (handles a model that
    # ignores the JSON instruction and free-lines its captions instead).
    lines = [ln.strip().lstrip("0123456789.-) ").strip() for ln in text.splitlines()]
    return [normalize_to_ascii(ln) for ln in lines if ln]


# ---------------------------------------------------------------------------
# Scene decoding: ASCII grid + token grid
# ---------------------------------------------------------------------------

def scene_to_ASCII(scene: list[list[int]], id_to_char: dict[int, str],
                   null_ids: frozenset[int] = frozenset()) -> list[str]:
    """
    Decode a 2D integer tile-id grid into a list of ASCII row strings using the tileset map.

    This is used only to build the captioning prompt; the integer grid itself is carried
    through to the output unchanged.

    Drop the leading all-null padding rows and render any remaining null cell as empty
    space ('-', the air char) so the LLM never sees the '@' void and invents a feature
    ("void", out of bounds, etc.) from it. This matches how deterministic_caption treats
    null tiles (open/non-terrain), keeping the grid the LLM reads consistent with the
    metadata it's grounded on.
    """
    first_real = next((r for r in range(len(scene)) if not all(t in null_ids for t in scene[r])), 0)
    return ["".join("-" if tile in null_ids else id_to_char[tile] for tile in row)
            for row in scene[first_real:]]


def build_char_to_token(id_to_char: dict[int, str]) -> dict[str, str]:
    """Map each tile character to a 'T<NN>' token, NN = its numeric tile ID."""
    width = max(2, len(str(max(id_to_char) if id_to_char else 0)))
    return {char: f"T{idx:0{width}d}" for idx, char in id_to_char.items()}


def build_token_dict_string(id_to_char: dict[int, str], char_to_token: dict[str, str],
                            names: dict[str, str]) -> str:
    """Render the 'T0x = Name' symbol dictionary used in the prompt for --grid-format tokens."""
    lines = []
    for idx in sorted(id_to_char):
        char = id_to_char[idx]
        token = char_to_token[char]
        name = names.get(char, char).split(':')[0].strip()
        lines.append(f"{token} = {name}")
    return "\n".join(lines)


def scene_to_tokens(scene: list[list[int]], char_to_token: dict[str, str], unknown_char: str = "?") -> str:
    """Render a scene as a space-separated grid of 'T<NN>' tokens instead of raw ASCII."""
    unknown = char_to_token.get(unknown_char, "T??")
    return "\n".join(
        " ".join(char_to_token.get(str(tile), unknown) for tile in row)
        for row in scene
    )


# Conservative chars-per-token ratios: the LOW end of the observed range for that prompt
# shape, so dividing by it overestimates the token count and the fitted context window
# errs toward leaving enough room. The space-separated 'T<NN>' token grid is far denser
# (~1.5-2.0 chars/token) than the character ASCII grid + prose (~3-4).
_CHARS_PER_TOKEN = {"tokens": 1.5, "ascii": 3.5}


def estimate_prompt_tokens(text: str, grid_format: str = "ascii") -> int:
    """Conservative (deliberately high) token estimate for an Ollama prompt."""
    return int(len(text) / _CHARS_PER_TOKEN.get(grid_format, 3.5)) + 1


def fit_num_ctx(prompt: str, floor_ctx: int, max_ctx: int, gen_tokens: int,
                grid_format: str = "ascii") -> tuple[int, bool]:
    """
    Pick an Ollama num_ctx that holds the prompt AND leaves room to generate.

    Grows floor_ctx up to max_ctx when a big prompt (e.g. the space-separated token grid
    on a wide level) would otherwise fill the whole window and starve generation -- the
    starvation that makes Ollama return an empty/truncated response. Returns
    (effective_ctx, fits); fits is False when even max_ctx cannot seat the prompt with
    meaningful room to answer.
    """
    prompt_tokens = estimate_prompt_tokens(prompt, grid_format)
    needed = prompt_tokens + gen_tokens + 512  # headroom for reprompt clauses
    effective = min(max(floor_ctx, needed), max_ctx)
    fits = prompt_tokens + 256 <= max_ctx
    return effective, fits


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str, char_to_id: dict[str, int]) -> list[tuple[list[list[int]], str, dict]]:
    """
    Load integer tile-id scenes for an LLM to caption.

    {path} could be:
        - a JSON file produced by a create_*_json_data.py script (or a previous captioning
          pass): a list of entries, each a dict with a "sample" key holding the 2D integer
          tile-id grid (a "scene" key or a bare grid are also accepted). Any other keys the
          entry carries -- metadata (source_level, scan_mode, data, ...) and captions from
          earlier sources (deterministic_captions, some_model_captions, ...) -- are returned
          as `attrs` so main() can copy them straight through to the output. This is what
          lets a captioned dataset be fed back in to accumulate sources instead of
          replacing them.

        - a directory of VGLC-ASCII level files (*.txt), one whole level per file. These
          are encoded to integer grids (via char_to_id) so the output format is the same
          either way. There is no metadata to carry, so `attrs` is empty.

    Returns a list of (scene, label, attrs) tuples, where each scene is a 2D grid of integer
    tile ids, label identifies the source, and attrs is the dict of all other input attributes
    (empty for directory input). The integer grid is what gets stored in the output; it is only
    decoded to ASCII/tokens transiently for the captioning prompt.
    """
    if os.path.isdir(path):
        levels = load_levels(path)
        files = sorted(Path(path).glob("*.txt"))
        return [([[char_to_id[c] for c in row] for row in level], file.name, {})
                for level, file in zip(levels, files)]

    with open(path, "r") as f:
        data = json.load(f)

    scenes = []
    for entry in data:
        if isinstance(entry, dict):
            grid = entry["sample"] if "sample" in entry else entry["scene"]
            attrs = {k: v for k, v in entry.items() if k not in ("sample", "scene")}
        else:
            grid = entry
            attrs = {}
        scenes.append((grid, Path(path).name, attrs))
    return scenes


# ---------------------------------------------------------------------------
# Checkpointing / resume (JSONL, crash-safe, shard-aware)
# ---------------------------------------------------------------------------

def default_checkpoint_path(output: str | None, shard_index: int, shard_count: int) -> str:
    """
    Derive the checkpoint (.jsonl) path from --output by simply swapping its ".json" extension
    for ".jsonl" -- this is not user-configurable, so a given --output always has one obvious,
    predictable checkpoint file next to it. Falls back to "captions.json" -> "captions.jsonl"
    when --output isn't given. Shard-suffixed (e.g. '.shard0of4') when running as part of a
    multi-machine split, so parallel shards never write over each other's checkpoint.
    """
    base = output or "captions.json"
    stem = base[:-5] if base.endswith(".json") else base
    suffix = f".shard{shard_index}of{shard_count}" if shard_count > 1 else ""
    return f"{stem}{suffix}.jsonl"


def load_checkpoint(path: str) -> dict[int, dict]:
    """
    Read a checkpoint jsonl file (if it exists) into {global_index: entry}. Each line is a
    JSON object carrying a private "_index" key (the scene's position in the full, unsharded
    dataset) alongside the same fields main() would otherwise put straight into --output.
    Corrupt trailing lines (e.g. a write cut off mid-flush by a crash) are skipped, not fatal,
    so resuming after a hard crash only ever loses the one in-flight scene, never prior work.
    """
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[entry["_index"]] = entry
    return done


class CheckpointWriter:
    """
    Append-only, thread-safe JSONL writer. Every completed scene is written and flushed to
    disk immediately (os.fsync included) so a crash, kill, or lab-machine reboot at any point
    loses at most the scene currently in flight -- never previously finished work. The lock is
    unused by this script's own (serial) captioning loop, but keeps this class safe to reuse
    from callers that write to it from several request-handling threads.
    """

    def __init__(self, path: str, resume: bool):
        self.path = path
        self._lock = threading.Lock()
        mode = "a" if resume and os.path.exists(path) else "w"
        self._f = open(path, mode, buffering=1)  # line-buffered

    def write(self, index: int, entry: dict) -> None:
        record = dict(entry)
        record["_index"] = index
        line = json.dumps(record)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()
            os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()


def resolve_resume(checkpoint_path: str, force_resume: bool, force_restart: bool) -> bool:
    """
    Decide whether this run should resume from an existing checkpoint or start fresh.

    - No checkpoint file present: nothing to decide, start fresh.
    - --force-resume: resume automatically, no prompt.
    - --force-restart: delete the existing checkpoint and start fresh, no prompt.
    - Otherwise (interactive default): ask the user, since silently resuming an old checkpoint
      or silently discarding it are both surprising things to do without asking.
    """
    if not os.path.exists(checkpoint_path):
        return False

    if force_resume:
        return True

    if force_restart:
        os.remove(checkpoint_path)
        print(f"[checkpoint] --force-restart: deleted existing {checkpoint_path}\n")
        return False

    while True:
        answer = input(
            f"Checkpoint file '{checkpoint_path}' already exists. Resume from it? [y/n]: "
        ).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            os.remove(checkpoint_path)
            print(f"[checkpoint] Starting fresh: deleted existing {checkpoint_path}\n")
            return False
        print("Please answer 'y' or 'n'.")


def finalize_output(checkpoint_path: str, output_path: str | None) -> list[dict]:
    """
    Rebuild the final captioned_dataset from the checkpoint (source of truth), in original
    scene order, and write it to --output if given. Because every run -- whether it finishes
    cleanly or is resumed after a crash -- reads back through this function, --output always
    reflects everything completed so far.
    """
    done = load_checkpoint(checkpoint_path)
    ordered = [done[i] for i in sorted(done)]
    for entry in ordered:
        entry.pop("_index", None)
    if output_path:
        with open(output_path, "w") as f:
            json.dump(ordered, f, indent=2)
        print(f"Captioned dataset saved to {output_path} ({len(ordered)} scene(s), from checkpoint {checkpoint_path})")
    return ordered


# ---------------------------------------------------------------------------
# Tileset helpers
# ---------------------------------------------------------------------------

def filter_tile_set(scene: str, tileset: dict) -> dict:
    """
    Given an ASCII/token level scene and the complete tile set for the game the given scene
    belongs to, return a filtered tile set dict to insert in the LLM prompt to conserve (a
    marginal amount of) tokens, and to avoid hallucination/confusion in the LLM response.
    This filtered tileset only contains the k: v pairs that are found in the provided scene.

    Args:
        scene (str): the ASCII level scene
        tileset (dict -- char: str): the complete ASCII char: string description tile set

    Returns:
        dict -- char: str: the filtered tile set that only contains tiles found in the level
    """
    filtered = {char: desc for char, desc in tileset.items() if char in scene}
    return filtered


# Tags that mark a tile as carrying no per-object information for the metadata's object-count
# block. This replaces the old hardcoded-per-game character list (e.g. {"P", "-", "t", "@", "#"})
# with a tag-driven check, so the same logic works for any registered game's tileset as long as
# its tiles are tagged consistently. See the accompanying chat message for exactly which tags
# each game's tileset needs to carry for this to behave correctly.
_METADATA_SKIP_TAGS = {"null", "empty", "ground", "spawn"}


def deterministic_caption(scene: list[list[int]], id_to_char: dict[int, str], char_to_id: dict[str, int],
                          tile_descriptors: dict, names: dict, describe_locations: bool = False,
                          describe_absence: bool = False, data: dict = None) -> str:
    """
    Build a block of pre-computed structural metadata for an integer tile-id scene, to feed
    the LLM as grounding context.

    The metadata is purely mechanical fact about the grid that the LLM can lean on instead of
    re-deriving it from the raw ASCII:
      - raw occupied-cell counts per object tile type (readable names)
      - terrain top-of-column heights (structural solid tiles, ignoring enemies/hazards)
      - floor / ceiling analysis (top and bottom rows)
      - left / center / right region column boundaries

    Args:
        scene: 2D grid of integer tile ids (the raw scene, not the ASCII decode).
        id_to_char, char_to_id, tile_descriptors: tileset maps from extract_tileset.
        names: the game's descriptive tileset dict (char -> readable description) used to render
            the object-count lines with human names.
        describe_locations / describe_absence / data: kept for signature compatibility with the
            caller; this metadata builder doesn't use them.

    Returns:
        A multi-line metadata string.
    """

    def is_null(tile: int) -> bool:
        return "null" in tile_descriptors.get(id_to_char.get(tile), set())

    first_real = next((r for r in range(len(scene)) if not all(is_null(t) for t in scene[r])), len(scene))
    scene = scene[first_real:] or scene

    height = len(scene)
    width = len(scene[0])

    # Terrain = structural solid tiles (ground, walls, blocks); enemies and hazards that
    # happen to be "solid" don't count as terrain.
    def is_terrain(tile: int) -> bool:
        desc = tile_descriptors.get(id_to_char.get(tile), set())
        return "solid" in desc and "enemy" not in desc and "hazard" not in desc

    # Object tile counts: raw occupied cells per type, by readable name, most common first.
    # Skip tiles tagged as null/empty/ground/spawn -- see _METADATA_SKIP_TAGS above.
    counts = Counter()
    for row in scene:
        for tile in row:
            char = id_to_char.get(tile)
            if char is None:
                continue
            desc = tile_descriptors.get(char, set())
            if desc & _METADATA_SKIP_TAGS:
                continue
            counts[char] += 1
    if counts:
        count_lines = "\n".join(
            f"  {names.get(char, char).split(':')[0].strip()}: {n}"
            for char, n in sorted(counts.items(), key=lambda kv: -kv[1])
        )
    else:
        count_lines = "  (none)"

    # Per-column ground profile: the elevation of the LOWEST standable floor surface, found by
    # scanning up from the bottom. A solid tile is a surface only if the two cells above it are
    # open, since the player is two tiles tall. Floating platforms higher up are ignored (the
    # lowest surface wins); the model reads the grid for those. A column with no standable
    # surface is reported as "wall" (mostly solid, blocked) or "pit" (no safe footing).
    def is_open(tile: int) -> bool:
        return "solid" not in tile_descriptors.get(id_to_char.get(tile), set())

    ground = []
    for c in range(width):
        surface = next(
            (r for r in range(height - 1, 1, -1)
             if is_terrain(scene[r][c]) and is_open(scene[r - 1][c]) and is_open(scene[r - 2][c])),
            None,
        )
        if surface is not None:
            ground.append(str((height - 1) - surface))
        else:
            solid = sum(1 for r in range(height) if is_terrain(scene[r][c]))
            ground.append("wall" if solid * 2 >= height else "pit")
    ground_line = " ".join(f"{tok:>4}" for tok in ground)

    # Ceiling (top row): solid coverage and contiguous gap count.
    def edge_summary(row: list[int]) -> str:
        solid = sum(1 for t in row if is_terrain(t))
        if solid == 0:
            return "absent"
        if solid == width:
            return "solid across"
        if solid * 2 < width:
            return "mostly open"
        gaps, in_gap = 0, False
        for t in row:
            if is_terrain(t):
                in_gap = False
            elif not in_gap:
                gaps += 1
                in_gap = True
        return f"present with {gaps} gap" + ("s" if gaps != 1 else "")

    ceiling = edge_summary(scene[0])

    left_end = width // 3
    center_end = 2 * width // 3
    regions = (f"left=cols 1-{left_end}, center=cols {left_end + 1}-{center_end}, "
               f"right=cols {center_end + 1}-{width}")

    return (
        "Object tile counts (raw occupied cells per type, one placed object may span several cells):\n"
        f"{count_lines}\n\n"
        f"Ground surface profile per column (columns 1-{width} left to right; each value is how "
        "many tiles the lowest standable floor sits above the bottom, 0=floor at the very bottom; "
        "'wall'=solid blocked column, 'pit'=no safe footing (an open drop, or a floor sealed off "
        "by hazards)):\n"
        f"  {ground_line}\n"
        f"Ceiling (top row): {ceiling}\n\n"
        "Region boundaries (use these when assigning left/center/right): "
        f"{regions}"
    )


# ---------------------------------------------------------------------------
# API key loading (replaces the old dotenv/.env approach)
# ---------------------------------------------------------------------------

def load_api_key(api_key_path: str) -> str:
    """Read an API key from the first line of a text file."""
    with open(api_key_path, "r", encoding="utf-8") as f:
        return f.readline().strip()


# ---------------------------------------------------------------------------
# Backend callers
# ---------------------------------------------------------------------------
# Each call_*() function has the same signature:
#   call_*(system_prompt, user_content, model, api_key, max_tokens, timeout, retries, **kw) -> str
# returning the raw text response. Adding a new backend (e.g. a future vision-capable model)
# means writing one more function with this signature and registering it in BACKEND_CALLERS.

def call_claude(system_prompt, user_content, model, api_key, max_tokens, timeout, retries, **_):
    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_content}],
    }).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                parts = result.get("content", [])
                return "".join(p.get("text", "") for p in parts).strip()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries - 1:
                wait = min(2 ** attempt * 5, 60)
                print(f"  [RETRY {attempt + 1}/{retries - 1}] {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Claude request failed after {retries} attempts: {e}") from e


def call_openai(system_prompt, user_content, model, api_key, max_tokens, timeout, retries, **_):
    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                choices = result.get("choices", [])
                if not choices:
                    return ""
                return (choices[0].get("message", {}).get("content") or "").strip()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries - 1:
                wait = min(2 ** attempt * 5, 60)
                print(f"  [RETRY {attempt + 1}/{retries - 1}] {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise RuntimeError(f"OpenAI request failed after {retries} attempts: {e}") from e


def call_gemini(system_prompt, user_content, model, api_key, max_tokens, timeout, retries, **_):
    # Gemini's generateContent has no separate system-turn field in this simple form, so fold
    # the system prompt into the single text block, same as the original MM gemini branch did.
    prompt = f"{system_prompt}\n\n{user_content}"
    payload = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens},
    }).encode("utf-8")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if not candidates:
                    return ""
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts).strip()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries - 1:
                wait = min(2 ** attempt * 5, 60)
                print(f"  [RETRY {attempt + 1}/{retries - 1}] {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Gemini request failed after {retries} attempts: {e}") from e


def call_ollama(system_prompt, user_content, model, api_key, max_tokens, timeout, retries,
                num_ctx=OLLAMA_NUM_CTX, temperature=0.4, **_):
    context = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    completion = ollama.chat(
        model=model,
        messages=context,
        think=False,
        options={"num_ctx": num_ctx, "temperature": temperature, "num_predict": max_tokens},
        # Keep the model resident in VRAM between calls instead of the ollama default
        # 5-minute unload, which would otherwise dwarf actual inference time.
        keep_alive="30m",
    )
    return (completion.message.content or "").strip()


BACKEND_CALLERS = {
    "claude": call_claude,
    "openai": call_openai,
    "gemini": call_gemini,
    "ollama": call_ollama,
    # "smolvlm": call_smolvlm,      # TODO if/when re-added
    # "moondream": call_ollama,     # (moondream/llava are just ollama models)
    # "llava": call_ollama,
}


# ---------------------------------------------------------------------------
# Captioning
# ---------------------------------------------------------------------------

def llm_caption(scene: str, deterministic: str, game: str = "Mega Man", tileset: dict = None,
                llm: str = "ollama", model: str = "qwen3.5:9b", num_captions: int = EXPECTED_CAPTIONS,
                *, vocab_extra: list[str] = (), rule_extra: list[str] = (),
                api_key: str = None, max_tokens: int = 2048, timeout: int = 120, retries: int = 5,
                num_ctx: int = OLLAMA_NUM_CTX, temperature: float = 0.4,
                max_caption_retries: int = MAX_CAPTION_RETRIES,
                retry_on_empty: bool = True, retry_on_nonascii: bool = True,
                max_reprompts: int = MAX_REPROMPTS,
                bad_chars: set = None) -> list[str]:
    """
    Prompt the selected backend for `num_captions` diverse captions of a scene, retrying on
    the wrong caption count (always) and, if enabled, on empty or non-ASCII responses.

    game: display name used in the prompt ("Mega Man", "Mario Maker 2", ...).
    vocab_extra / rule_extra: this game's prompt plug-ins (see GAMES[...]["prompt_vocab"] /
        ["prompt_rules"] in util/descriptive_tilesets.py).
    bad_chars: optional set the caller can pass in and reuse across scenes, so the "banned
        character" list accumulates over an entire run rather than resetting every scene.
        If omitted, a fresh set is used for just this call.
    Returns a list of caption strings (possibly fewer than num_captions if every retry budget
    is exhausted).
    """
    if bad_chars is None:
        bad_chars = set()

    system_prompt = build_system_prompt(num_captions, game, vocab_extra=vocab_extra, rule_extra=rule_extra)
    caption_reminder = build_caption_reminder(num_captions)

    tileset_str = json.dumps(tileset or {}, indent=2)
    deterministic_msg = (
        "For grounding and reference, here is pre-computed structural metadata for this level. Treat it as "
        "accurate, stay consistent with it, and don't invent features it doesn't support or "
        "re-count terrain from the grid yourself. How to read it:\n"
        "- Object tile counts are RAW occupied-cell counts, NOT object counts. One placed thing "
        "can span many cells, so a large count usually means one big feature, not many separate "
        "ones. Use rough quantities, never exact tile numbers.\n"
        "- Ground surface profile per column gives the level's walkable floor shape: each value is "
        "how high the lowest standable floor sits in that column (rising values = steps/hills, a "
        "flat run = flat ground), while 'wall' marks a solid blocked column and 'pit' marks a column "
        "with no safe footing (an open drop, or a floor sealed off by hazards). It only tracks the "
        "lowest floor, so read the grid for raised platforms or overhead structures above it. This output "
        "should shape your classification of the ground, not simply the tiles at the very bottom row. "
        "Keep in mind that the player always moves left-to-right in non-vertical segments, so base "
        "your ascending vs descending structure analysis based on this. DO NOT explicitly mention 'ground levels' "
        "in your captions; use the data to inform your natural captions.\n"
        "- Ceiling describes the top row's overhead terrain; region boundaries map columns to "
        "left/center/right.\n"
        "Still write the captions in your own words per the rules above:\n"
        f"{deterministic}"
    )

    user_content_base = (
        f"Here is the tile set for {game}:\n{tileset_str}\n\n"
        f"Level Scene:\n{scene}\n\n"
        f"{deterministic_msg}\n\n"
        f"{caption_reminder}"
    )

    call_fn = BACKEND_CALLERS.get(llm)
    if call_fn is None:
        raise ValueError(f"Unknown --llm backend: {llm!r}. Available: {sorted(BACKEND_CALLERS)}")

    captions: list[str] = []
    attempt = 0            # empty / non-ASCII reprompts
    caption_retries = 0    # wrong-count reprompts

    while attempt < max_reprompts and caption_retries < max_caption_retries:
        active_content = (
            user_content_base
            + (build_avoidance_clause(bad_chars) if retry_on_nonascii else "")
            + build_count_clause(num_captions, caption_retries)
        )

        raw = call_fn(system_prompt, active_content, model, api_key, max_tokens, timeout, retries,
                     num_ctx=num_ctx, temperature=temperature)
        captions = parse_captions(raw)

        if not captions and retry_on_empty:
            attempt += 1
            print(f"[reprompt {attempt}/{max_reprompts}] empty/unparseable response, retrying...")
            continue
        if not captions:
            break  # retry_on_empty disabled: accept the empty result rather than loop

        if retry_on_nonascii:
            new_bad = find_non_ascii_chars(captions)
            if new_bad:
                bad_chars.update(new_bad)
                attempt += 1
                print(f"[reprompt {attempt}/{max_reprompts}] non-ASCII char(s) "
                      f"{', '.join(new_bad)!r} in caption, retrying...")
                captions = []
                continue

        if len(captions) != num_captions:
            caption_retries += 1
            print(f"[caption retry {caption_retries}/{max_caption_retries}] got {len(captions)} "
                  f"caption(s), expected {num_captions}; retrying...")
            captions = []
            continue

        break

    if len(captions) > num_captions:
        captions = captions[:num_captions]
    return captions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    argparser = argparse.ArgumentParser(
        description="Caption VGLC-ASCII levels for any registered game with an LLM."
    )

    argparser.add_argument("--levels", default="../TheVGLC/MegaMan/Enhanced",
                           help="JSON dataset of integer tile-id scenes, or a directory of "
                                "VGLC-ASCII level .txt files")
    argparser.add_argument("--game", default="MM-Full", choices=list(GAMES),
                           help="Which game/tileset to caption for. Selects the descriptive "
                                "tileset, the prompt game name, the tileset JSON used to decode "
                                "integer scenes, and this game's prompt vocab/rules (from "
                                "util/descriptive_tilesets.py::GAMES).")
    argparser.add_argument("--llm", choices=list(BACKEND_CALLERS), default="ollama",
                           help="The source of the LLM inference used to caption the provided level scenes.")
    argparser.add_argument("--model", default=None,
                           help="Model to prompt for captions. When omitted, defaults per --llm branch "
                                "(see DEFAULT_MODELS).")
    argparser.add_argument("--api-key-file", default=None,
                           help="Path to a text file containing the API key (on its first line). "
                                "Required for --llm claude/openai/gemini; ignored for ollama.")
    argparser.add_argument("--output", default=None,
                           help="Optional path to write the captioned [{scene, caption}] list as JSON")
    argparser.add_argument("--limit", type=int, default=None,
                           help="Max number of scenes to caption. Defaults to the entire dataset")
    argparser.add_argument("--num_captions", type=int, default=EXPECTED_CAPTIONS,
                           help=f"Number of captions each LLM query is expected to return. Defaults to {EXPECTED_CAPTIONS}")
    argparser.add_argument("--show_captions", action="store_true",
                           help="Print each scene's generated captions to the console as they're produced. "
                                "When omitted, a tqdm progress bar is shown instead.")
    argparser.add_argument("--caption-mode", default="keyed", choices=["legacy", "keyed"],
                           help="Output schema. 'keyed' (default) writes the captions as a list under "
                                "--caption-key (default '<model>_captions'). 'legacy' writes "
                                "'caption'/'caption1'/... fields instead.")
    argparser.add_argument("--caption-key", default=None,
                           help="Key to store the caption list under when --caption-mode keyed. "
                                "Defaults to '<model>_captions'.")

    # --- reprompt controls (on by default) ---
    argparser.add_argument("--no-retry-on-empty", dest="retry_on_empty", action="store_false",
                           help="Disable reprompting on an empty/unparseable LLM response "
                                "(enabled by default).")
    argparser.add_argument("--no-retry-on-nonascii", dest="retry_on_nonascii", action="store_false",
                           help="Disable reprompting when a caption contains non-ASCII characters "
                                "(enabled by default).")
    argparser.add_argument("--max-reprompts", type=int, default=MAX_REPROMPTS,
                           help=f"How many empty/non-ASCII reprompts to tolerate per scene. Default {MAX_REPROMPTS}.")
    argparser.add_argument("--max-caption-retries", type=int, default=MAX_CAPTION_RETRIES,
                           help=f"How many wrong-caption-count reprompts to tolerate per scene. Default {MAX_CAPTION_RETRIES}.")
    argparser.set_defaults(retry_on_empty=True, retry_on_nonascii=True)

    # --- grid rendering / context window management ---
    argparser.add_argument("--grid-format", choices=["ascii", "tokens"], default="ascii",
                           help="How the level grid is rendered in the prompt. 'ascii' (default) uses "
                                "raw tile characters. 'tokens' renders each cell as a 'T<NN>' token, "
                                "which some models count/parse more reliably.")
    argparser.add_argument("--num-ctx", type=int, default=OLLAMA_NUM_CTX,
                           help=f"Baseline Ollama context window (tokens); grown per-scene up to "
                                f"--max-num-ctx when needed. Default {OLLAMA_NUM_CTX}. Ollama only.")
    argparser.add_argument("--max-num-ctx", type=int, default=OLLAMA_MAX_NUM_CTX,
                           help=f"Ceiling for the per-scene Ollama context window. Default {OLLAMA_MAX_NUM_CTX}.")
    argparser.add_argument("--temperature", type=float, default=0.4,
                           help="Ollama sampling temperature. Default 0.4.")
    argparser.add_argument("--max-tokens", type=int, default=2048,
                           help="Max generation tokens per LLM call. Default 2048.")
    argparser.add_argument("--timeout", type=int, default=120,
                           help="Per-request timeout in seconds (API backends). Default 120.")
    argparser.add_argument("--retries", type=int, default=5,
                           help="Retry attempts on network failure (API backends). Default 5.")

    # --- incremental checkpointing / resume ---
    resume_group = argparser.add_mutually_exclusive_group()
    resume_group.add_argument("--force-resume", action="store_true",
                           help="If the checkpoint .jsonl already exists, resume from it automatically "
                                "without prompting.")
    resume_group.add_argument("--force-restart", action="store_true",
                           help="If the checkpoint .jsonl already exists, delete it and start completely "
                                "fresh, without prompting.")

    # --- splitting one dataset across multiple machines ---
    argparser.add_argument("--shard-index", type=int, default=0,
                           help="This machine's shard number (0-based) when splitting the dataset across "
                                "several machines. Combine with --shard-count.")
    argparser.add_argument("--shard-count", type=int, default=1,
                           help="Total number of machines splitting the dataset. 1 (default) means no sharding.")

    args = argparser.parse_args()

    if args.llm in ("claude", "openai", "gemini") and not args.api_key_file:
        argparser.error(f"--api-key-file is required for --llm {args.llm}")

    return args


def main() -> list[list[str]]:

    args = parse_args()

    # Resolve the selected game: its human-readable name (for the prompt), its descriptive
    # tileset dict (char -> readable description), its tileset JSON, and this game's prompt
    # plug-ins.
    game = GAMES[args.game]
    game_name = game["name"]
    tile_names = game["tiles"]["tiles"]
    tileset_path = game["tileset"]
    prompt_vocab = game.get("prompt_vocab", [])   # TODO per game -- see chat message
    prompt_rules = game.get("prompt_rules", [])   # e.g. MM2's multi-tile-object dedup note

    _, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)

    null_ids = frozenset(tid for tid, ch in id_to_char.items() if "null" in tile_descriptors.get(ch, set()))

    scenes = load_dataset(args.levels, char_to_id)
    scenes = scenes[:args.limit]

    indexed_scenes = list(enumerate(scenes))
    if args.shard_count > 1:
        indexed_scenes = [(i, s) for i, s in indexed_scenes if i % args.shard_count == args.shard_index]

    model = args.model or DEFAULT_MODELS[args.llm]
    api_key = load_api_key(args.api_key_file) if args.api_key_file else None
    caption_key = args.caption_key or f"{model}_captions"
    llmstr = f"{args.llm} - {model}"

    # Token-grid rendering setup (only built if requested, to avoid the extra work otherwise).
    char_to_token = build_char_to_token(id_to_char) if args.grid_format == "tokens" else None

    checkpoint_path = default_checkpoint_path(args.output, args.shard_index, args.shard_count)
    resume = resolve_resume(checkpoint_path, args.force_resume, args.force_restart)

    already_done = load_checkpoint(checkpoint_path) if resume else {}
    if already_done:
        print(f"[resume] {len(already_done)} scene(s) already captioned in {checkpoint_path}; skipping those.\n")
        indexed_scenes = [(i, s) for i, s in indexed_scenes if i not in already_done]

    writer = CheckpointWriter(checkpoint_path, resume=resume)

    start_time = time.time()
    print(f"[timing] Captioning started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
    print(f"[checkpoint] Writing incremental progress to {checkpoint_path} "
          f"({len(indexed_scenes)} scene(s) to caption this run)\n")

    # Shared across the whole run so the "banned character" list accumulates rather than
    # resetting every scene.
    run_bad_chars: set = set()

    caption_lists = []

    try:
        for i, (scene, label, attrs) in tqdm(indexed_scenes, total=len(indexed_scenes), desc="Captioning",
                                              unit="scene", disable=args.show_captions):

            scene_str = "\n".join(scene_to_ASCII(scene, id_to_char, null_ids))
            if args.grid_format == "tokens":
                grid_for_prompt = scene_to_tokens(scene, char_to_token)
            else:
                grid_for_prompt = scene_str

            filtered_tiles = filter_tile_set(scene_str, tile_names)

            det_caption = deterministic_caption(scene, id_to_char, char_to_id, tile_descriptors, names=tile_names)

            # Context-window sizing only matters for the local Ollama backend.
            num_ctx = args.num_ctx
            if args.llm == "ollama":
                probe_prompt = grid_for_prompt + json.dumps(filtered_tiles) + det_caption
                num_ctx, fits = fit_num_ctx(probe_prompt, args.num_ctx, args.max_num_ctx,
                                            args.max_tokens, args.grid_format)
                if not fits:
                    msg = (f"[skip] {label}: prompt needs ~{estimate_prompt_tokens(probe_prompt, args.grid_format)} "
                           f"tokens, more than --max-num-ctx ({args.max_num_ctx}) can hold with room to "
                           f"generate. Use --grid-format ascii or raise --max-num-ctx.")
                    print(msg) if args.show_captions else tqdm.write(msg)
                    continue

            caption_set = llm_caption(
                grid_for_prompt, game=game_name, model=model, tileset=filtered_tiles, llm=args.llm,
                deterministic=det_caption, num_captions=args.num_captions,
                vocab_extra=prompt_vocab, rule_extra=prompt_rules,
                api_key=api_key, max_tokens=args.max_tokens, timeout=args.timeout, retries=args.retries,
                num_ctx=num_ctx, temperature=args.temperature,
                max_caption_retries=args.max_caption_retries,
                retry_on_empty=args.retry_on_empty, retry_on_nonascii=args.retry_on_nonascii,
                max_reprompts=args.max_reprompts, bad_chars=run_bad_chars,
            )

            if args.show_captions:
                print(f"------------------ [{llmstr}]  [{label}] (index {i}) ------------------\n")
                for j, caption in enumerate(caption_set):
                    print(f"[Caption {j + 1}/{len(caption_set)}] {caption}\n")

            if len(caption_set) != args.num_captions:
                skip_msg = (f"[skip] {label}: got {len(caption_set)} caption(s) instead of "
                            f"{args.num_captions}; skipping this scene.\n")
                print(skip_msg) if args.show_captions else tqdm.write(skip_msg)
                continue

            caption_lists.append(caption_set)
            entry = dict(attrs)
            entry["scene"] = scene
            if args.caption_mode == "legacy":
                entry["caption"] = caption_set[0]
                for idx, caption in enumerate(caption_set[1:], start=1):
                    entry[f"caption{idx}"] = caption
                entry["model"] = llmstr
            else:
                entry[caption_key] = caption_set

            writer.write(i, entry)
    finally:
        writer.close()

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[timing] Captioning finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
    print(f"[timing] Total captioning time: {elapsed:.2f}s ({elapsed / 60:.2f} min)\n")

    finalize_output(checkpoint_path, args.output)

    if args.shard_count > 1:
        print(f"[shard] This is shard {args.shard_index}/{args.shard_count}. Once every shard has finished, "
              f"combine all shard checkpoints with: python merge_shards.py <checkpoint-glob> --output <final.json>\n")

    return caption_lists


if __name__ == "__main__":
    main()