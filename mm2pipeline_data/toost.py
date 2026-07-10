"""Batch-convert SMM2 .bcd level files to JSON and PNG images via toost.exe.

toost.exe lives in toost_stuff/bin/ and loads its sprite/font assets relative to
its working directory (toost_stuff/), so the subprocess is run with that folder
as cwd. All input/output paths are resolved to absolute first, so conversion
works regardless of where this module is invoked from.

If the input folder carries a level_metadata.json index (written by
mm2pipeline_data.extract), each level's server-side difficulty/tags are folded into
the exported JSONs, since Toost can't emit fields that aren't in the .bcd.
"""
import os
import sys
import json
import shutil
import argparse
import subprocess

from . import paths


def print_success(text): print(f"\033[92m[+] {text}\033[0m")
def print_info(text):    print(f"\033[94m[*] {text}\033[0m")
def print_warn(text):    print(f"\033[93m[-] {text}\033[0m")
def print_error(text):   print(f"\033[91m[!] {text}\033[0m")

EXE_NAME = "toost.exe" if sys.platform == "win32" else "toost"

# toost.exe and its assets are bundled under MM2_Files/toost_stuff/.
TOOST_DIR = paths.TOOST_DIR

# Toost seemingly needs a font for it to work here. Less than ideal, but we can workaround.
# The NotoSansJP-Bold.otf font isn't necessary since we can fallback to a system font.
# Suggested by claude, this should not require any admin priveleges and should work on non-windows as well.
FONT_PATH = TOOST_DIR / "fonts" / "NotoSansJP-Bold.otf"
SYS_FONTS = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
FONT_FALLBACK = [
    os.path.join(SYS_FONTS, "arial.ttf"),
    os.path.join(SYS_FONTS, "segoeui.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # linux
    "/Library/Fonts/Arial.ttf",                         # macOS
]


def ensure_font():
    """Make sure the font toost.exe loads at startup exists, sourced from
    the system instead of committed to the repo."""

    if FONT_PATH.is_file():
        return True
    for src in FONT_FALLBACK:
        if os.path.isfile(src):
            FONT_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, FONT_PATH)
            print_info(f"Sourced toost font from the system: {src} -> {FONT_PATH.name}")
            return True
    print_info("No system font was found to source. Copy any .otf/.ttf there and rename it to {FONT_PATH.name}.") #This should literally never happen but just in case.
    return False


def find_exe():
    candidates = [
        os.path.join("bin", EXE_NAME),
        os.path.join(".", EXE_NAME),
        TOOST_DIR / "bin" / EXE_NAME,
        TOOST_DIR / EXE_NAME,
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    found = shutil.which(EXE_NAME)
    if found:
        return found
    return None


def asset_cwd(exe_path):
    """Directory toost.exe should run in so it finds img/ and fonts/.

    The exe sits in toost_stuff/bin/, but its assets are one level up in
    toost_stuff/, so when the exe is inside a 'bin' folder use its parent."""
    exe_dir = os.path.dirname(exe_path)
    if os.path.basename(exe_dir).lower() == "bin":
        return os.path.dirname(exe_dir)
    return exe_dir


def world_size(json_path):
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("object_count", 0) + data.get("ground_count", 0)
    except Exception:
        return 0


def load_metadata_index(input_dir):
    # level_metadata.json (from mm2pipeline_data.extract) maps each .bcd stem to the
    # server-side fields that aren't in the .bcd payload: {difficulty, tags}.
    path = os.path.join(input_dir, "level_metadata.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print_warn(f"Could not read metadata index '{path}': {e}")
        return {}


def attach_metadata(json_path, meta):
    # Toost doesn't know about tags or difficulty (they aren't in the .bcd), so
    # fold them into the JSON alongside the fields Toost did decode.
    if not os.path.isfile(json_path):
        return
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        data["tags"] = meta.get("tags", [])
        data["difficulty"] = meta.get("difficulty")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print_warn(f"Could not attach metadata to '{json_path}': {e}")


def batch_convert(exe, input_dir, output_dir, images_dir, min_objects,
                  remove_grid, objects_over_pipes):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    images_dir = os.path.abspath(images_dir)
    run_cwd = asset_cwd(exe)

    bcd_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".bcd")]
    if not bcd_files:
        print_info(f"No .bcd files found in: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    metadata_index = load_metadata_index(input_dir)
    print_info(f"Processing {len(bcd_files)} file(s) -> {output_dir} (json) / {images_dir} (images)")
    if metadata_index:
        print_info(f"Loaded metadata for {len(metadata_index)} level(s) from level_metadata.json")
    print("-" * 60)

    ok = skipped = failed = 0
    for filename in sorted(bcd_files):
        bcd_path  = os.path.join(input_dir, filename)
        stem      = os.path.splitext(filename)[0]
        ow_json   = os.path.join(output_dir, f"{stem}_overworld.json")
        sub_json  = os.path.join(output_dir, f"{stem}_subworld.json")
        ow_png    = os.path.join(images_dir, f"{stem}_overworld.png")
        sub_png   = os.path.join(images_dir, f"{stem}_subworld.png")

        cmd = [exe, "-p", bcd_path,
               "--overworldJson", ow_json, "--subworldJson", sub_json,
               "-o", ow_png, "-s", sub_png]

        if remove_grid:
            cmd.append("-r")
        if objects_over_pipes:
            cmd.append("-e")

        print(f"  {filename} ...", end=" ", flush=True)
        result = subprocess.run(cmd, capture_output=True, cwd=run_cwd)
        if result.returncode != 0:
            print("\033[91mFAILED\033[0m")
            for stream in (result.stdout, result.stderr):
                if stream:
                    print(f"    {stream.decode(errors='replace').strip()}")
            failed += 1
            continue

        # Metadata applies to the whole level: overworld always, subworld if kept.
        meta = metadata_index.get(stem)
        if meta is not None:
            attach_metadata(ow_json, meta)

        # Remove subworld JSON/PNG if it's below the size threshold
        sub_size = world_size(sub_json)
        if sub_size < min_objects:
            os.remove(sub_json)
            if os.path.exists(sub_png):
                os.remove(sub_png)
            print(f"\033[92mOK\033[0m  \033[93m(subworld skipped: {sub_size} objects)\033[0m")
            skipped += 1
        else:
            if meta is not None:
                attach_metadata(sub_json, meta)
            print("\033[92mOK\033[0m")

        ok += 1

    print("-" * 60)
    print_success(f"Done: {ok}/{len(bcd_files)} converted, {skipped} empty subworlds dropped, {failed} failed.")


def main(argv=None):
    os.system("color")

    parser = argparse.ArgumentParser(description="Batch convert SMM2 .bcd level files to JSON and PNG images.")
    parser.add_argument("--input",              required=True, help="Folder containing .bcd files")
    parser.add_argument("-o", "--output_folder", help="JSON output folder (default: <input>/json/)")
    parser.add_argument("--images-output",      help="PNG output folder (default: <folder>/images/)")
    parser.add_argument("--min-objects",        type=int, default=1,
                        help="Minimum object+ground count to keep a subworld (default: 1)")
    parser.add_argument("--remove-grid",        action="store_true", help="Render without grid")
    parser.add_argument("--objects-over-pipes", action="store_true", help="Render objects over pipes")
    args = parser.parse_args(argv)

    exe = find_exe()
    if not exe:
        print_error(f"Could not find '{EXE_NAME}'.")
        print_info(f"Expected it at {TOOST_DIR / 'bin' / EXE_NAME} (or on PATH).")
        print_info("Build it from https://github.com/TheGreatRambler/toost in an "
                   "MSYS2 MinGW64 terminal with: mingw32-make BUILD=release")
        sys.exit(1)
    print_success(f"Using exe: {exe}")
    if not ensure_font():
        sys.exit(1)

    output_dir = args.output_folder or os.path.join(args.input, "json")
    images_dir = args.images_output or os.path.join(args.input, "images")
    batch_convert(exe, args.input, output_dir, images_dir,
                  min_objects=args.min_objects,
                  remove_grid=args.remove_grid,
                  objects_over_pipes=args.objects_over_pipes)


if __name__ == "__main__":
    main()
