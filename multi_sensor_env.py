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
def zip_3dicts(statusd, battd, diffd):
    full_state = {k:(v, battd[k], diffd[k]) for k,v in statusd.items()}
    return full_state    
batt_evolution = functools.partial(sensor_env.battery_dynamics,10)
def what_is_noop(state):
    status = state[0]
    return 0 if status == 0 else 1
def get_new_state(old_state, action):
    action_num,action_val = action
    action_key = 'S'+str(action_num)
    #print('getting reward,{}:{}'.format(state, reward))
    full_actions = {k:what_is_noop(old_state[k]) for k in old_state
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
    def __init__(self, num_sensors=2):
        self.action_space = spaces.Tuple((spaces.Discrete(num_sensors),spaces.Discrete(2)))
        base_state = spaces.Tuple((spaces.Discrete(3),spaces.Discrete(11)))
        obs_basis = {'S'+str(i):base_state for i in range(num_sensors)}
        self.observation_space = spaces.Dict(obs_basis)
        self.base_state = {k:(0,10) for k in obs_basis}
        self.state = {k:(0,10) for k in obs_basis}
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
        reset_state = self.base_state
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
        self.action_space = spaces.Tuple((spaces.Discrete(1),spaces.Discrete(2)))
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
        old_state = self.state
        reward = get_reward(old_state)
        #print('getting reward,{}:{}'.format(state, reward))
        new_state = get_new_state(old_state, action)
        #print('action: ', action, full_actions, self.state, new_state)
        self.state = new_state
        return new_state, reward, False, {}
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




