import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from src.analysis.sensitivity_curve import plotSensitivity, save
from src.analysis.results import loadResults
from src.utils.model import loadExperiment

exp_paths = sys.argv[1:]

ax = plt.axes()
bounds = []
for exp_path in exp_paths:
    exp = loadExperiment(exp_path)
    results = loadResults(exp)

    bound = plotSensitivity(results, 'lambda', ax)
    bounds.append(bound)

lower = min(map(lambda x: x[0], bounds))
upper = max(map(lambda x: x[1], bounds))

ax.set_ylim([lower, upper])

save(exp, 'lambda_sensitivity', trial=0)
# plt.show()
