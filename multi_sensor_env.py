import gym
from gym import spaces
from gym.utils import seeding
import sensor_env
import random
import string
import json
import os
import csv
import sqlite3
import functools
def zip_dicts(statusd, battd):
    full_state = {k:(v, battd[k]) for k,v in statusd.items()}
    return full_state
batt_evolution = functools.partial(sensor_env.battery_dynamics,10)
def get_new_state(old_state, action):
    action_num,action_val = action
    action_key = 'S'+str(action_num)
    #print('getting reward,{}:{}'.format(state, reward))
    full_actions = {k:0 for k in old_state
                           if k != action_key}
    full_actions[action_key] = action_val

    new_statuses = {k: sensor_env.status_dynamics(v[0],v[1],full_actions[k])
                        for k,v in old_state.items()}
    new_batteries = {k:batt_evolution(v[0],v[1]) for
                         k, v in old_state.items()} 
    new_state = zip_dicts(new_statuses, new_batteries)
    return new_state
def get_reward(old_state):
    individual_rewards = [sensor_env.get_reward(v) for k,v in old_state.items()]
    awake_reward = int(any(individual_rewards))
    capable_reward = int(all([v[1] for k,v in old_state.items()]))
    #print('awake,{}, capable,{}'.format(awake_reward, capable_reward))
    return (awake_reward and capable_reward)
class MultiSensorEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Tuple((spaces.Discrete(2),spaces.Discrete(3)))
        self.observation_space = spaces.Dict({'S0':spaces.Tuple((spaces.Discrete(3),
                                                   spaces.Discrete(11)))
                                             ,'S1':spaces.Tuple((spaces.Discrete(3),
                                                   spaces.Discrete(11)))
                                             })
        self.state = {'S0':(0,10), 'S1':(0,10)}
        self.fname = os.getcwd()+'/tmp/'+''.join(random.choice(string.ascii_lowercase) for _ in range(5))+'.json'
        with open(self.fname, 'a+') as f:
            json.dump({'data':[]},f)
        self.record = []
        self.seed()
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self, action):
        """action is like {'S1':0}"""
        assert self.action_space.contains(action)
        old_state = self.state
        reward = get_reward(old_state)
        #print('getting reward,{}:{}'.format(state, reward))
        new_state = get_new_state(old_state, action)
        #print('action: ', action, full_actions, self.state, new_state)
        self.state = new_state
        return new_state, reward, False, {}
    def reset(self):
        reset_state = {'S0':(0,10), 'S1':(0,10)}
        self.state = reset_state
        with open(self.fname, 'r') as f:
            previous = json.load(f)
        if len(previous['data'])>3:
            previous['data'].pop(0)
        previous['data'].append(self.record)
        with open(self.fname, 'w') as f:
            json.dump(previous,f)
        self.record = []
        return reset_state
    def render(self):
        self.record.append(self.state)
class TestSensorEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Tuple((spaces.Discrete(1),spaces.Discrete(3)))
        self.observation_space = spaces.Dict({'S0':spaces.Tuple((spaces.Discrete(3),
                                                   spaces.Discrete(6)))
                                             })
        self.state = {'S0':(0,5)}
        self.fname = os.getcwd()+'/tmp/'+''.join(random.choice(string.ascii_lowercase) for _ in range(5))+'.json'
        with open(self.fname, 'a+') as f:
            json.dump({'data':[]},f)
        self.record = []
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
        new_statuses = {k: sensor_env.status_dynamics(v[0],v[1],full_actions[k])
                        for k,v in state.items()}
        new_batteries = {k:sensor_env.battery_dynamics(v[0],v[1]) for
                         k, v in state.items()} 
        if min(new_batteries.values()) == 0:
            done=False
        else: 
            done=False
        new_state = zip_dicts(new_statuses, new_batteries)
        self.state= new_state
        return new_state, reward, done, {}
    def reset(self):
        reset_state = {'S0':(0,5)}
        self.state = reset_state
        with open(self.fname, 'r') as f:
            previous = json.load(f)
        if len(previous['data'])>3:
            previous['data'].pop(0)
        previous['data'].append(self.record)
        with open(self.fname, 'w') as f:
            json.dump(previous,f)
        self.record = []
        return reset_state
    def render(self):
        self.record.append(self.state)




