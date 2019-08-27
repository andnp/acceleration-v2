import sys
import json

CODES = ['000', '001', '010', '011', '100', '101', '110', '111']

example = sys.argv[1]
path = '/'.join(example.split('/')[:-1])

f_name = example.split('/')[-1]

agent, code = f_name[:-5].split('_')

with open(example, 'r') as f:
    d = f.read()

for c in CODES:
    new_agent = f'{agent}_{c}'
    new = d.replace(f'{agent}_{code}', new_agent)

    with open(path + '/' + new_agent + '.json', 'w') as f:
        f.write(new)
