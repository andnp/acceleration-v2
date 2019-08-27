from src.agents.agents import getAgent

class StepModel():
    def __init__(self, data):
        self.step = data['step']
        self.reward = data['reward']

class EpisodeModel():
    def __init__(self, data):
        self.steps = data['steps']
        self.total_reward = data['total_reward']

class BaseProblem:
    def __init__(self, exp, idx):
        self.Agent = getAgent(exp.agent)

        # what parameter permutation should we use
        perm = exp.getPermutation(idx)

        # add gamma to the agent parameters
        # gamma is a problem parameter, but the agent needs access to it
        self.metaParameters = perm['metaParameters']
        self.metaParameters['gamma'] = self.getGamma()

    def getEnvironment(self):
        raise NotImplementedError()

    def getRepresentation(self):
        raise NotImplementedError()

    def getAgent(self):
        raise NotImplementedError()

    def getGamma(self):
        raise NotImplementedError()

    def getSteps(self):
        raise NotImplementedError()

    def sampleExperiences(self):
        raise NotImplementedError()

    def evaluateStep(self, step):
        pass

    def evaluateEpisode(self, episode):
        pass
