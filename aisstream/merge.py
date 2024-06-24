import os
import json
from pathlib import Path

INPUT_FOLDER = Path("data-unfiltered")
OUTPUT_FOLDER = Path("merged")
OUTPUT_FILE = "merged.json"

all_paths = []

for file in os.listdir(INPUT_FOLDER):
    if file.startswith("position"):
        with open(INPUT_FOLDER / file, "r") as f:
            position_data = json.load(f)

        paths = [
            [(p["Latitude"], p["Longitude"], p["Sog"], p["Cog"]) for p in path]
            for path in position_data.values()
        ]
        all_paths.extend(paths)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
with open(OUTPUT_FOLDER / OUTPUT_FILE, "w") as f:
    json.dump(all_paths, f)
