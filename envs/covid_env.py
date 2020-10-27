import gym
from gym import spaces
import covasim as cv

N_ACTIONS = 1
DAILY_TESTS = 100

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
        # self.observation_space =

    def get_observations(self):
        return

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
        obs = None
        return None, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.sim.initialize()
        return

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return
