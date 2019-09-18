import sys
import json
import glob
import os
sys.path.append(os.getcwd())

from src.utils.path import fileName, up

prob_folder = sys.argv[1]
new_prob = sys.argv[2]
jsons = glob.glob(f'{prob_folder}/**/*.json', recursive=True)

old_prob = fileName(prob_folder)

for json in jsons:
    with open(json, 'r') as f:
        d = f.read()

        new = d.replace(old_prob, new_prob)
        new_path = json.replace(old_prob, new_prob)

        os.makedirs(up(new_path), exist_ok=True)
        with open(new_path, 'w') as f:
            f.write(new)
