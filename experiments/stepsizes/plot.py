import glob
import subprocess

experiments = glob.glob('experiments/stepsizes/*/*')
experiments = [f for f in experiments if not '.' in f]

md_file = open('experiments/stepsizes/plots.md', 'w')

def last(path):
    parts = path.split('/')
    return parts[len(parts) - 1]

for exp in experiments:
    jsons = exp + '/*.json'

    if last(exp) == 'td' or '_h' in last(exp):
        subprocess.run(f'python experiments/stepsizes/single_stepsize_combinations.py {jsons}', stdout=subprocess.PIPE, shell=True)
    else:
        subprocess.run(f'python experiments/stepsizes/rmsve_rmspbe_combinations.py {jsons}', stdout=subprocess.PIPE, shell=True)

    exp_path = exp.replace('experiments/stepsizes', '.')
    md_file.write(f'## {exp_path}\n')
    md_file.write(f'![]({exp_path}/plots/rmsve_rmspbe_square.svg)\n')
    md_file.flush()
