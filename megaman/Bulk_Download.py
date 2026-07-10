import requests
import gzip
import os
import sys
import time
import argparse
import json

from tqdm import tqdm

# Import the repo-wide constant for the master metadata sidecar. This script lives in
# megaman/, so the repo root (which holds the util package) isn't on sys.path by default.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util.common_settings as common_settings


parser = argparse.ArgumentParser()
parser.add_argument("--target", type=int, default=100, help="Number of valid levels to download")
parser.add_argument("--start_id", type=int, default=200000, help="The level ID of the starting point of the bulk download (higher = more recent)")
parser.add_argument("--force", action="store_true", help="Re-download and overwrite levels that already exist locally instead of skipping them")
parser.add_argument("--show_downloads", action="store_true", help="Print the original per-level status lines (Downloaded/Already exists/No level found/etc.) instead of the default tqdm progress bar")
args = parser.parse_args()


def status(msg):
    """Emit a routine per-attempt status line: shown only with --show_downloads, otherwise the
    progress bar conveys progress. Routed through tqdm.write so it never clobbers an active bar."""
    if args.show_downloads:
        tqdm.write(msg)

TARGET_DOWNLOADS = args.target


SAVE_DIR = os.path.expandvars(
    r"%LOCALAPPDATA%\MegaMaker\Levels"
)
os.makedirs(SAVE_DIR, exist_ok=True)


LOG_FILE = os.path.join(SAVE_DIR, "download_log.txt")

# Machine-readable sidecar of the same metadata, keyed by level ID as a string. This is the
# single master file the training-data pipeline reads (create_megaman_json_data.py), so the
# level's name/author/downloads/likes/dislikes can be attached to every generated sample
# without re-hitting the API. It lives at a constant, global path in the repo (not next to
# the downloaded .mmlv files) so there is exactly one copy. Kept in sync with the log below.
METADATA_FILE = common_settings.MEGAMAN_METADATA_PATH
os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)

# IMPORTANT: NEVER overwrite existing log
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Mega Man Successfully Downloaded Levels\n")
        f.write("=" * 60 + "\n\n")

# Load any existing metadata so repeat runs accumulate instead of clobbering.
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    metadata = {}

downloaded = 0
failed = 0
level_id = args.start_id

# Progress toward the target number of valid downloads. The loop scans many level IDs that get
# skipped (missing, low downloads, bad rating), so the bar tracks successful downloads, not
# iterations; set_postfix surfaces the id currently being scanned and the failure count so the
# bar still shows life during long stretches with no new download. --verbose disables the bar
# and restores the original per-level status prints.
pbar = tqdm(total=TARGET_DOWNLOADS, desc="Downloading levels", unit="level", disable=args.show_downloads)

while downloaded < TARGET_DOWNLOADS:

    pbar.set_postfix(scanning=level_id, failed=failed)

    filename = os.path.join(SAVE_DIR, f"{level_id}.mmlv")

    already_downloaded = os.path.exists(filename)
    if already_downloaded and not args.force:
        status(f"Already exists: {level_id}")
        level_id += 1
        continue

    try:
        
        info_url = f"https://api.megamanmaker.com/level/{level_id}"
        info_response = requests.get(info_url, timeout=10)

        if info_response.status_code != 200:
            status(f"No level found: {level_id}")
            failed += 1
            level_id += 1
            continue

        info = info_response.json()

        name = info.get("name", "")
        author = info.get("authorName", "")
        downloads = info.get("downloads", 0)
        likes = info.get("likes", 0)
        dislikes = info.get("dislikes", 0)

        
        if downloads < 5:
            status(f"Not enough downloads: {level_id}")
            level_id += 1
            continue

        if likes < dislikes:
            status(f"bad rating: {level_id}")
            level_id += 1
            continue

        
        meta_url = f"https://api.megamanmaker.com/level/download/{level_id}"
        response = requests.get(meta_url, timeout=10)

        if response.status_code != 200:
            status(f"download API failed: {level_id}")
            failed += 1
            level_id += 1
            continue

        meta = response.json()

        if "location" not in meta:
            status(f"Not valid ID: {level_id}")
            failed += 1
            level_id += 1
            continue

        
        compressed = requests.get(meta["location"], timeout=10).content
        level_data = gzip.decompress(compressed).decode()

        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(level_data)

        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"Level ID: {level_id}\n")
            f.write(f"Name: {name}\n")
            f.write(f"Author: {author}\n")
            f.write(f"Downloads: {downloads}\n")
            f.write(f"Likes: {likes}\n")
            f.write(f"Dislikes: {dislikes}\n")
            f.write("Status: downloaded\n")
            f.write("-" * 60 + "\n\n")

        # Mirror the same fields into the machine-readable sidecar (keyed by str id so it
        # matches the .mmlv/.txt filename stem the pipeline uses as the lookup key), and
        # rewrite it now so an interrupted run keeps everything downloaded so far.
        metadata[str(level_id)] = {
            "name": name,
            "author": author,
            "downloads": downloads,
            "likes": likes,
            "dislikes": dislikes,
        }
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Forced overwrites of already-present levels are updates, not new downloads, so
        # they don't count toward the target.
        if already_downloaded:
            status(f"Updated: {level_id}")
        else:
            downloaded += 1
            pbar.update(1)
            status(f"Downloaded: {level_id} ({downloaded}/{TARGET_DOWNLOADS})")

    except Exception as e:
        # Errors are rare and worth surfacing even in bar mode, so route them through tqdm.write
        # (rather than status()) so they show regardless of --show_downloads.
        tqdm.write(f"[ERROR] {level_id} → {e}")
        failed += 1

    level_id += 1
    time.sleep(0.5)

pbar.close()

print("\nFinished")
print("Downloaded:", downloaded)
print("Failed:", failed)