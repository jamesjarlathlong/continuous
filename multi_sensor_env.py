import gym
from gym import spaces
from gym.utils import seeding
import sensor_env
def zip_dicts(statusd, battd):
    full_state = {k:(v, battd[k]) for k,v in statusd.items()}
    return full_state
class MultiSensorEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Tuple((spaces.Discrete(2),spaces.Discrete(3)))
        self.observation_space = spaces.Dict({'S0':spaces.Tuple((spaces.Discrete(3),
                                                   spaces.Discrete(11)))
                                             ,'S1':spaces.Tuple((spaces.Discrete(3),
                                                   spaces.Discrete(11)))
                                             })
        self.state = {'S0':(0,10), 'S1':(0,10)}
        self.seed()
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self, action):
        """action is like {'S1':0}"""
        assert self.action_space.contains(action)
        action_num,action_val = action
        action_key = 'S'+str(action_num)
        state = self.state
        individual_rewards = [sensor_env.get_reward(v) for k,v in state.items()]
        reward = int(any(individual_rewards))
        full_actions = {k:0 for k in self.state
                           if k != action_key}
        full_actions[action_key] = action_val
        
        new_statuses = {k: sensor_env.status_dynamics(*v,full_actions[k])
                        for k,v in state.items()}
        new_batteries = {k:sensor_env.battery_dynamics(*v) for
                         k, v in state.items()} 
        if min(new_batteries.values()) == 0:
            done=True
            print('DONE')
        else: 
            done=False
        new_state = zip_dicts(new_statuses, new_batteries)
        print('actions: {}, old:{}, new:{}'.format(full_actions, state, new_state))
        self.state= new_state
        return new_state, reward, done, {}
    def reset(self):
        reset_state = {'S0':(0,10), 'S1':(0,10)}
        self.state = reset_state
        return reset_state