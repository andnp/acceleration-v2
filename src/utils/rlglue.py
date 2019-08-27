from RlGlue import BaseAgent

def identity(s):
    return s

class OffPolicyWrapper(BaseAgent):
    def __init__(self, agent, b_policy, t_policy, observationChannel = identity):
        self.agent = agent
        self.b_policy = b_policy
        self.t_policy = t_policy
        self.observationChannel = observationChannel
        self.s_t = None
        self.a_t = None
        self.obs_t = None

    def start(self, s):
        self.s_t = s
        self.obs_t = self.observationChannel(s)
        self.a_t = self.b_policy.selectAction(s)
        return self.a_t

    def step(self, r, s):
        gamma = self.agent.gamma
        obs_tp1 = self.observationChannel(s)
        p = self.t_policy.ratio(self.b_policy, self.s_t, self.a_t)
        self.agent.update(self.obs_t, self.a_t, obs_tp1, r, gamma, p)

        self.s_t = s
        self.a_t = self.b_policy.selectAction(s)
        self.obs_t = obs_tp1

        return self.a_t

    def end(self, r):
        p = self.t_policy.ratio(self.b_policy, self.s_t, self.a_t)
        self.agent.update(self.obs_t, self.a_t, self.obs_t, r, 0, p)
        self.agent.reset()

class GvfHordeWrapper(BaseAgent):
    def __init__(self, Agent, features, params, policies, gammas, rewards, b_policy, observationChannel = identity):
        num_agents = len(policies)
        if len(gammas) != num_agents or len(rewards) != num_agents:
            raise Exception('Must have equal number of gammas, rewards, and policies')

        self.agents = [ Agent(features, params) for _ in range(num_agents) ]
        self.gammas = gammas
        self.policies = policies
        self.rewards = rewards
        self.b_policy = b_policy
        self.observationChannel = observationChannel
        self.s_t = None
        self.a_t = None
        self.obs_t = None

    def start(self, s):
        self.s_t = s
        self.obs_t = self.observationChannel(s)
        self.a_t = self.b_policy.selectAction(s)
        return self.a_t

    def _updateAll(self, obs_tp1):
        for i, agent in enumerate(self.agents):
            gamma = self.gammas[i](self.s_t, self.a_t)
            t_policy = self.policies[i]
            r = self.rewards[i](self.s_t, self.a_t)
            
            p = t_policy.ratio(self.b_policy, self.s_t, self.a_t)
            
            agent.update(self.obs_t, self.a_t, obs_tp1, r, gamma, p)

    def step(self, _, s):
        obs_tp1 = self.observationChannel(s)
        
        self._updateAll(obs_tp1)

        self.s_t = s
        self.a_t = self.b_policy.selectAction(s)
        self.obs_t = obs_tp1

        return self.a_t

    def end(self, r):
        self._updateAll(self.obs_t)
        for agent in self.agents:
            agent.reset()