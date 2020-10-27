import gym
from gym import spaces
import covasim as cv
import numpy as np

N_ACTIONS = 1
DAILY_TESTS = 100

class Feature():
    def __init__(self, name, discrete, ob_dim=2, low=0, high=1):
        self.name = name
        self.discrete = discrete
        self.ob_dim = ob_dim
        self.low = low
        self.high = high

    def get_space(self):
        if self.discrete:
            return spaces.Discrete(self.ob_dim)
        else:
            return spaces.Box(low=self.low, high=self.high)

FEATURES = {
    Feature('age', False, low=0, high=np.inf),
    Feature('death_prob', low=0, high=1),
    Feature('exposed', True, ob_dim=2),
    Feature('susceptible', True, ob_dim=2),
    Feature('sex', True, ob_dim=2),
    Feature('symp_prob', False, low=0, high=1),
    Feature('tested', True, ob_dim=2),
}

class CovidEnv(gym.Env):
    """Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CovidEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.sim = self.sim = cv.Sim()
        # Example when using discrete actions:
        self.action_space = spaces.Continuous(N_ACTIONS)

        spaces = [feature.get_space() for feature in FEATURES]
        self.observation_space = spaces.Tuple(spaces)

    def get_observations(self):
        obs = [self.sim.people[feature.name] for feature in FEATURES]
        obs = np.asarray(obs).T
        return obs

    def step(self, action):
        # Execute one time step within the environment
        assert self.action_space.contains(action), f"{action} ({type(action)} invalid "
        self.sim['interventions'] = test_num(
            daily_tests=DAILY_TESTS,
            symp_test=action,
        )
        self.sim.step()
        reward = -self.sim.check_death()
        done = self.sim.complete
        info = None
        obs = self.get_observations()
        return obs, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.sim.initialize()
        return

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return
