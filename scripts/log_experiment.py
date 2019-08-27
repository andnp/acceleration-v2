import os
import re
import sys
import glob
import subprocess
from datetime import datetime
from PyExpUtils.models.Config import getConfig

exp_dir = getConfig().experiment_directory

if not os.path.isfile(f'{exp_dir}/log.md'):
    f = open(f'{exp_dir}/log.md', 'w+')
    f.write('# Experiment Log\n')
    f.close()

with open(f'{exp_dir}/log.md', 'r') as f:
    lines = f.readlines()

exp_path = sys.argv[1]
exp_name = exp_path.split(exp_dir + '/')[1]

commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8')[:-1]

trial = 0
for f in glob.glob(f'{exp_path}/trials/*'):
    cur_trial = int(re.findall(r'\d+', f)[0]) + 1
    if cur_trial > trial:
        trial = cur_trial

entry = f"""\
 * **experiment**: {exp_name}
 * **trial**: {trial}
 * **commit**: {commit}
"""

current_time = datetime.now()
date = current_time.strftime('%m/%d/%y')
i = 0
for l in lines:
    i += 1
    if date in l:
        break

if i == len(lines):
    lines.insert(2, f'**{date}**\n')
    lines.insert(3, f'{entry}\n')
else:
    lines.insert(i, entry)
    lines.insert(i + 1, '---\n')

new_log = "".join(lines)

with open(f'{exp_dir}/log.md', 'w') as f:
    f.write(new_log)
