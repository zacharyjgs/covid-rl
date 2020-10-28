import gym
from gym import spaces
import covasim as cv
import numpy as np

N_ACTIONS = 1
DAILY_TESTS = [100] * 60
POP_SIZE = 20000
POP_INFECTED = 10
TRACE_PROB = 0.9
TRACE_TIME = 1.0
QUAR_PERIOD = 14

class Feature():
    def __init__(self, name, discrete=True, ob_dim=2, low=0, high=1, shape=(POP_SIZE, 1)):
        self.name = name
        self.discrete = discrete
        self.ob_dim = ob_dim
        self.low = low
        self.high = high
        self.shape = shape

    def get_space(self):
        if self.discrete:
            return spaces.MultiDiscrete([self.ob_dim] * POP_SIZE)
        else:
            return spaces.Box(low=self.low, high=self.high, shape=self.shape)

FEATURES = {
    Feature('age', discrete=False, low=0, high=np.inf),
    Feature('death_prob', discrete=False, low=0, high=1),
    Feature('exposed', discrete=True, ob_dim=2),
    Feature('susceptible', discrete=True, ob_dim=2),
    Feature('sex', discrete=True, ob_dim=2),
    Feature('symp_prob', discrete=False, low=0, high=1),
    Feature('tested', discrete=True, ob_dim=2),
}

class CovidEnv(gym.Env):
    """Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CovidEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.sim = cv.Sim(
            pop_size=POP_SIZE,
            pop_infected=POP_INFECTED
        )
        self.sim.initialize()
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([np.inf]))

        observation_spaces = [feature.get_space() for feature in FEATURES]
        obs_dim = np.sum([obs_space.shape[0] for obs_space in observation_spaces])
        self.observation_space = spaces.Tuple(observation_spaces)
        self.observation_space.shape = [obs_dim]

        self.dead = 0

        self.trace_probs = {k: TRACE_PROB for k in self.sim.people.layer_keys()}
        self.trace_time = {k: TRACE_TIME for k in self.sim.people.layer_keys()}

    def get_observations(self):
        obs = [self.sim.people[feature.name] for feature in FEATURES]
        obs = np.concatenate(obs)
        return obs

    def step(self, action):
        # Execute one time step within the environment
        assert self.action_space.contains(action), f"{action} ({type(action)} invalid "
        symp_test = action[0]
        self.sim['interventions'] = [
            cv.interventions.test_num(
                daily_tests=DAILY_TESTS,
                symp_test=symp_test
            ),
            cv.interventions.contact_tracing(
                trace_probs=self.trace_probs, 
                trace_time=self.trace_time,
                quar_period=QUAR_PERIOD
            ),
        ]
        self.sim.step()
        self.sim.people.is_exp = self.sim.people.true('exposed') 
        new_dead = np.sum(self.sim.people.dead)
        reward = self.dead - new_dead
        self.dead = new_dead
        done = self.sim.complete
        info = None
        obs = self.get_observations()
        return obs, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.dead = 0
        self.sim.initialize(reset=True)
        obs = self.get_observations()
        return obs

    # def render(self, mode='human', close=False):
    #     # Render the environment to the screen
    #     return
