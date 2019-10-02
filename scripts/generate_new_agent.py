import sys
import json
import glob
import os
sys.path.append(os.getcwd())

from src.utils.path import fileName, up

old_agent_path = sys.argv[1]
problems = sys.argv[2:]

problems = filter(lambda p: '.' not in p and 'plots' not in p, problems)

with open(old_agent_path, 'r') as f:
    old_agent = f.read()

old_agent_base_name = fileName(up(old_agent_path))
old_agent_name = fileName(old_agent_path).replace('.json', '')
old_problem = fileName(up(up(old_agent_path)))

for problem in problems:
    path = f'{problem}/{old_agent_base_name}/{old_agent_name}.json'
    new = old_agent.replace(old_problem, fileName(problem))

    os.makedirs(up(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(new)
