# off-policy-acceleration

## Setting up repo
**This codebase only works with python 3.6 and above.**

Packages are stored in a `requirements.txt` file (standard for python codebases).
To install:
```
pip install -r requirements.txt
```
If `pip` is set to `python 2`, then you might need to do:
```
pip3 install -r requirements.txt
```

On machines that you do not have root access to (like compute canada machines), you will need to install in the user directory.
You can do this with:
```
pip install --user -r requirements.txt
```

## Step-by-step example
Here is a quick guide to run an already existent experiment on compute canada.
```bash
ssh $cedar
cd path/to/acceleration-v2
git pull # make sure you are up to date

# remove any old results that you might have lying around
# that way you don't accidentally zip them up and re-download them after the experiment
rm -rf results &

# check the cluster parameters
# make sure to balance using many parallel cpu cores
# while also being a good citizen of the resources (e.g. don't schedule 1000s of 2m jobs)
nano clusters/cedar.json

# run the experiment
python scripts/slurm_runs.py clusters/cedar.json ./ 100 experiments/myExperiment/*.json

# wait for a while
# then zip and download results
tar -czvf results.tar.gz results

# go back to your laptop
exit
scp $cedar:~/path/to/acceleration-v2/results.tar.gz ./
tar -xvf results.tar.gz

# plot your results
python analysis/learning_curve.py experiments/myExperiment/*.json
```

## Organization Patterns

### Experiments
All experiments are described as completely as possible within static data files.
I choose to use `.json` files for human readability and because I am most comfortable with them.
These are stored in the `experiments` folder, usually in a subdirectory with a short name for the experiment being run (e.g. `experiments/idealH` would specify an experiment that tests the effects of using h*).

Experiment `.json` files look something like:
```jsonc
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
```jsonc
{
    "account": "which compute canada account to use",
    "time": "how much time the job is expected to take",
    "nodes": "the number of cpu cores to use",
    "memPerCpu": "how much memory one parameter setting requires", // doesn't need to change
    "tasksPerNode": "how many parameter settings to run in serial on each cpu core"
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

## Entry Points
There are usually several different entry points for one of my codebases.
The entry points usually encode a different experiment, and they may not always be compatible with each other.

Here is a list of the entry points at the time of writing:
 - **src/runs.py** - This is the primary entry point. Running this will generate an `errors_summary.npy` result file for a single set of parameters over N runs.
 - **src/update_variance.py** - This entry point will compute the variance of the updates for any method. It will generate a `variance_summary.npy` result file for a single set of parameters over N runs. **This requires the Problem to have a `sampleExperiences` method.**
 - **src/ideal_h.py** - This is the latest and least tested entry point. It will run an average-over-runs experiment using the precomputed h*. It generates an `errors_summary.npy` result file for a single set of parameters over N runs. **This requires the Problem to have `R` and `P` defined.**

Each of the entry points will also have its own high-level scripts for running on compute-canada or locally.
These are a pain to maintain and work with right now, and they require a stupid amount of code-duplication (sorry about that!).
They are a work in progress.
