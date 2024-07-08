import re
import os
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import make_interp_spline as make_spline


INPUT_FOLDER = Path("data-unfiltered")
OUTPUT_FOLDER = Path("merged")
OUTPUT_FILE = "merged.json"

MAX_ALLOWED_SPEED = 30
MIN_REQUIRED_PATH_LENGTH = 20
MAX_MINUTES_BETWEEN_MESSAGES = 2 * 60
FIXED_SAMPLE_RATE_IN_MINUTES = 1

STATUS_ANCHORED = 1
STATUS_MOORED = 5
STATUSES_TO_SKIP = [STATUS_ANCHORED, STATUS_MOORED]

total_pre_filter = 0
total_post_filter = 0
all_paths = []


def strtime_to_unix_time(s):
    sec_decimal_start, sec_decimal_end = next(re.finditer("\.\d+", s)).span()
    decimals = s[sec_decimal_start:sec_decimal_end]
    s = s[:sec_decimal_start] + decimals[: min(len(decimals), 7)] + s[sec_decimal_end:]
    t = time.strptime(s, "%Y-%m-%d %H:%M:%S.%f %z %Z")
    return time.mktime(t)


def split_path(path):
    paths = [[path[0]]]
    for p1, p2 in zip(path, path[1:]):
        t1 = strtime_to_unix_time(p1["time_utc"])
        t2 = strtime_to_unix_time(p2["time_utc"])
        if t2 - t1 > MAX_MINUTES_BETWEEN_MESSAGES * 60:
            paths.append([])
        paths[-1].append(p2)
    return paths


def fix_sample_rate(path):
    x = np.array([strtime_to_unix_time(p[-1]) for p in path])
    y = np.array([p[:2] for p in path])
    print(len(x), len(set(x)))
    spline = make_spline(x, y)
    new_points = np.array(
        [
            spline(t)
            for t in range(int(x[0]), int(x[-1]), FIXED_SAMPLE_RATE_IN_MINUTES * 60)
        ]
    )
    return new_points


for file in os.listdir(INPUT_FOLDER):
    if file.startswith("position"):
        with open(INPUT_FOLDER / file, "r") as f:
            position_data = json.load(f)

        for path in position_data.values():
            total_pre_filter += len(path)

            path = filter(lambda p: p["Sog"] < MAX_ALLOWED_SPEED, path)
            path = filter(
                lambda p: p["NavigationalStatus"] not in STATUSES_TO_SKIP, path
            )
            path = list(path)

            if len(path) > 0:
                paths = split_path(path)
                for path in paths:
                    if len(path) > MIN_REQUIRED_PATH_LENGTH:
                        path = [
                            (
                                p["Latitude"],
                                p["Longitude"],
                                p["Sog"],
                                p["Cog"],
                                p["time_utc"],
                            )
                            for p in path
                        ]
                        path = fix_sample_rate(path)

                        total_post_filter += len(path)
                        all_paths.append(path)

print(f"{total_pre_filter=}")
print(f"{total_post_filter=}")
print(f"{len(all_paths)=}")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
with open(OUTPUT_FOLDER / OUTPUT_FILE, "w") as f:
    json.dump(all_paths, f)
