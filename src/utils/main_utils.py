import json
import os
from tqdm import tqdm


def load_standard_data(path, is_print=True):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if is_print:
        print(f"Loaded {len(data)} records from {path}")
    return data


def write_standard_data(data, path, makedirs=False):
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
