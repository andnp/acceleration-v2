# off-policy-acceleration

## Organization Patterns

### Experiments
All experiments are described as completely as possible within static data files.
I choose to use `.json` files for human readability and because I am most comfortable with them.
These are stored in the `experiments` folder, usually in a subdirectory with a short name for the experiment being run (e.g. `experiments/idealH` would specify an experiment that tests the effects of using h*).

Experiment `.json` files look something like:
```json
{
    "agent": "name of your agent (e.g. gtd2)",
    "problem": "name of the problem you're solving (e.g. randomwalk_inverted)",
    "metaParameters": { // <-- a dictionary containing all of the meta-parameters for this particular algorithm
        "alpha": [1, 0.5, 0.25], // <-- sweep over these 3 values of alpha
        "beta": 1.0, // <-- don't sweep over beta, always use 1.0
        "use_ideal_h": true,
        "lambda": [0.0, 0.1]
    }
}
```

### Problems
I define a **problem** as a combination of:
1) environment
2) representation
3) target/behavior policies
4) number of steps
5) gamma
6) starting conditions for the agent (like in Baird's)

The problem also ends up being a catch-all for any global variables (like error metrics, or sample generation for variance, or P for idealH, etc.).
This really sucks and needs to be cleaned up, but live and learn.

### results
The results are saved in a path that is defined by the experiment definition used.
The configuration for the results is specified in `config.json`, but we should never need to change that.
Using the current `config.json` yields results paths that look like:
```
<base_path>/results/<experiment short name>/<agent name>/<parameter values>/errors_summary.npy
```
Where `<base_path>` is defined when you run an experiment.

### src
This is where the source code is stored.
The only `.py` files it contains are "top-level" scripts that actually run an experiment.
No utility files or shared logic at the top-level.

**agents:** contains each of the agents that we are using.
For this project, the agents will likely all inherit the `BaseTD` agent.

**analysis:** contains shared utility code for analysing the results.
This *does not* contain scripts for analysing results, only shared logic.

**environments:** contains minimal implementations of just the environment dynamics.

**problems:** contains all of the various problem settings that we want to run.

**representations:** contains classes for generating fixed representations.
These are meant to be used in an online fashion, and do not need to be saved to file.

**utils:** various utility code snippets for doing things like manipulating file paths or getting the last element of an array.
These are just reusable code chunks that have no other clear home.
I try to sort them into files that roughly name how/when they will be used (e.g. things that manipulate files paths goes in `paths.py`, things that manipulate arrays goes in `arrays.py`, etc.).

### clusters
This folder contains the job submission information that is needed to run on a cluster.
These are also `.json` files that look like:
```json
{
    "account": "which compute canada account to use",
    "time": "how much time the job is expected to take",
    "nodes": "the number of cpus to use",
    "memPerCpu": "how much memory one parameter setting requires", // doesn't need to change
    "tasksPerNode": "how many parameter settings to run in serial on each node"
}
```
The only thing that really needs to change are `time` and `tasksPerNode`.
I try to keep jobs at about 1hr, so if running the code for one parameter setting takes 5 minutes, I'll set `tasksPerNode = 10` (I always leave a little wiggle room).

## Running the code
There are a few layers for running the code.
The most simple layer is directly running a single experiment for a single parameter setting.
The highest layer will schedule jobs on a cluster (or on a local computer) that sweeps over all of the parameter settings.

The higher layers of running the code work by figuring out how to call the most simple layer many times, then generating a script that calls the simple layer for each parameter setting.

**Everything should be run from the root directory of the repo!**

### Directly run experiment
Let's say you want to generate a learning curve over N runs of an algorithm.
```
python src/runs.py <N> <path/to/experiment.json> <parameter_setting_idx>
```
I want to note that it isn't super easy to know which `parameter_setting_idx` to use.
It is more simple to make an experiment description `.json` that only contains one possible parameter permutation (i.e. has no arrays in it).

This will save the results in the results folder as specified above.

These experiments are generally fast enough to run directly on your laptop.

### Run parameter sweeps
If you want to run a larger experiment (i.e. a parameter sweep), you'll want to run these on a cluster (like cedar).
```
python scripts/slurm_runs.py ./clusters/cedar.json <path/where/results/are/saved> <num runs> <path/to/experiment.json>
```
**example:** if I want to run an experiment called `./experiments/idealH/gtd2_not.json`
```
python scripts/slurm_runs.py ./clusters/cedar.json ./ 100 ./experiments/idealH/gtd2_not.json
```

To run multiple experiments at once, you can specify several `.json` files.
```
python scripts/slurm_runs.py ./clusters/cedar.json ./ 100 ./experiments/idealH/*.json
```
or
```
python scripts/slurm_runs.py ./clusters/cedar.json ./ 100 ./experiments/idealH/gtd2.json ./experiments/idealH/gtd2_not.json
```

### Generate learning curves
The top-level `analysis` folder contains the scripts for generating learning curves.
These are a bit more complicated; I'll fill this part of the readme out later.
For now, either (a) just trust them or (b) come bug Andy about them :)

```
python analysis/learning_curve.py <path/to/experiments.json>
```

**example:** One algorithm (one line)
```
python analysis/learning_curve.py ./experiments/idealH/gtd2_not.json
```

**example:** compare algorithms (multiple lines)
```
python analysis/learning_curve.py ./experiments/idealH/gtd2_not.json ./experiments/idealH/gtd2.json
```
