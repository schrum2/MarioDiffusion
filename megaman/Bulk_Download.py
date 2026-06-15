import requests
import gzip
import os
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--target", type=int, default=100, help="Number of valid levels to download")
args = parser.parse_args()

TARGET_DOWNLOADS = args.target


SAVE_DIR = os.path.expandvars(
    r"%LOCALAPPDATA%\MegaMaker\Levels"
)
os.makedirs(SAVE_DIR, exist_ok=True)


LOG_FILE = os.path.join(SAVE_DIR, "download_log.txt")

# IMPORTANT: NEVER overwrite existing log
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Mega Man Successfully Downloaded Levels\n")
        f.write("=" * 60 + "\n\n")

downloaded = 0
failed = 0
level_id = 200000


while downloaded < TARGET_DOWNLOADS:

    filename = os.path.join(SAVE_DIR, f"{level_id}.mmlv")

    if os.path.exists(filename):
        print(f"Already exists: {level_id}")
        level_id += 1
        continue

    try:
        
        info_url = f"https://api.megamanmaker.com/level/{level_id}"
        info_response = requests.get(info_url, timeout=10)

        if info_response.status_code != 200:
            print(f"No level found: {level_id}")
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
            print(f"Not enough downloads: {level_id}")
            level_id += 1
            continue

        if likes < dislikes:
            print(f"bad rating: {level_id}")
            level_id += 1
            continue

        
        meta_url = f"https://api.megamanmaker.com/level/download/{level_id}"
        response = requests.get(meta_url, timeout=10)

        if response.status_code != 200:
            print(f"download API failed: {level_id}")
            failed += 1
            level_id += 1
            continue

        meta = response.json()

        if "location" not in meta:
            print(f"Not valid ID: {level_id}")
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

        downloaded += 1
        print(f" Downloaded: {level_id} ({downloaded}/{TARGET_DOWNLOADS})")

    except Exception as e:
        print(f"[ERROR] {level_id} → {e}")
        failed += 1

    level_id += 1
    time.sleep(0.5)

print("\nFinished")
print("Downloaded:", downloaded)
print("Failed:", failed)