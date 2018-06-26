import gym
from gym import spaces
from gym.utils import seeding
def get_reward(state):
    status, battery = state
    if status == 0 and battery>0:
        reward = 1
    else:
        reward = 0
    return reward
def status_dynamics(status, battery, action):
    if action == 0:#noop
        if status == 1:#presleep
            new_status = 2#sleep
        else:
            new_status = status#same as the old status
    elif action == 1:#go to sleep
        new_status = 1 if status==0 else status
    elif action == 2:#wakeup
        new_status = 0#awake
    if status == 1:
        new_status = 2
    return new_status
def battery_dynamics(status, battery):
    if status == 2:#sleeping
        new_battery = min(battery+1, 5)
    else:#either pre-sleep or awake
        new_battery = max(0, battery-1)
    return new_battery
class SensorEnv(gym.Env):
    """Simple sleeping sensor environment
    Battery has 10 values, Status has 3 values
    0: On
    1: PreSleep
    2: Sleep
    If the Sensor remains on, you win a reward of 1.
    """
    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(11)))
        self.state = (0,10)
        self.seed()
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self, action):
        assert self.action_space.contains(action)
        status, battery = self.state
        reward = get_reward((status, battery))
        new_status = status_dynamics(status,battery, action)
        new_battery = battery_dynamics(status, battery)
        if new_battery == 0:
            done=True
        else: 
            done=False
        self.state= (new_status, new_battery)
        return (new_status, new_battery), reward, done, {}
    def reset(self):
        reset_state = (0,10)
        self.state = reset_state
        return reset_state
