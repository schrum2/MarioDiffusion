import json

IN_PATH = r"datasets\MM_TEST.json"
OUT_PATH = r"datasets\MM_TEST_clean.json"

HAZARD_ID = 19

with open(IN_PATH) as f:
    data = json.load(f)

clean = []
removed = 0
for entry in data:
    grid = entry.get("scene") or entry.get("sample")
    floor_row = grid[-1]
    hazard_pct = floor_row.count(HAZARD_ID) / len(floor_row)

    if hazard_pct > 0.7:
        removed += 1
        continue
    clean.append(entry)

print(f"original: {len(data)}")
print(f"removed: {removed}")
print(f"remaining: {len(clean)}")

with open(OUT_PATH, "w") as f:
    json.dump(clean, f, indent=2)

print(f"saved to {OUT_PATH}")