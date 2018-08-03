import gym
from gym import spaces
from gym.utils import seeding
import random_fields
import solar_getter
import functools
import sensor_env
import itertools
import numpy as np
import pandas as pd
import json
import os
import random
import string
import csv
import sys
import multi_sensor_env
def battery_dynamics(generated_power,battery_capacity, maxbatt, status, scaledbattery):
    battery = scaledbattery*(battery_capacity/maxbatt)
    discharge_voltage = 3.7 #Volts
    timeperiod = 0.5 #hours
    added_power =  generated_power*1000*timeperiod #mWh - generated is avg power in a timeperiod
    
    max_possible = battery_capacity*discharge_voltage #mAh e.g. 2000mAh times 3.7 volts = 7400mW 
    on_power = 56+45+5#mAh
    off_power = 0.5#mAh
    if status == 2:#sleeping
        used_power = (off_power*discharge_voltage*timeperiod)
    else:#either pre-sleep or awake
        used_power = (on_power*timeperiod*discharge_voltage)
    balance = added_power - used_power
    new_battery = min(battery+balance, battery_capacity)
    new_battery = max(new_battery,0)
    #print('added power in mWh: {}, used_power: {}, battery:{}, new_battery:{}'.format(added_power, used_power, battery, new_battery))
    normalised = new_battery*(maxbatt/battery_capacity)
    return int(round(normalised))
def runner(reducingseries,f):
    def wrapper(*args, **kwargs):
        current = reducingseries.pop(0)
        return f(current, *args, **kwargs)
    return wrapper
def slicer(wattage, start, duration):
    stop = start+duration
    return list(itertools.islice(itertools.cycle(wattage), start, stop))
def random_perturber(timeseries):
    psd_fun = functools.partial(random_fields.power_spectrum, 100)
    perturbation = random_fields.gpu_gaussian_proc(psd_fun, size=len(timeseries), scale =2)
    perturbation +=1 #perturbation around self value
    return np.multiply(timeseries, perturbation)
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
converter = lambda v: float(v) if is_number(v) else v
def random_start_generator():
    deltat = 0.5#hours
    num_steps = 365*24/deltat
    return np.random.randint(0, num_steps)
def get_new_state(batt_fun, battery_capacity, max_batt, old_state, action):
    action_num,action_val = action
    action_key = 'S'+str(action_num)
    #print('getting reward,{}:{}'.format(state, reward))
    full_actions = {k:multi_sensor_env.what_is_noop(old_state[k]) for k in old_state
                           if k != action_key}
    full_actions[action_key] = action_val

    new_statuses = {k: sensor_env.status_dynamics(v[0],v[1],full_actions[k])
                        for k,v in old_state.items()}
    new_batteries = {k:batt_fun(battery_capacity, max_batt, v[0],v[1]) for
                         k, v in old_state.items()} 
    new_state = multi_sensor_env.zip_dicts(new_statuses, new_batteries)
    return new_state
class SolarSensorEnv(gym.Env):
    """Simple sleeping sensor environment
    Battery has 10 values, Status has 3 values
    0: On
    1: PreSleep
    2: Sleep
    If the Sensor remains on, you win a reward of 1.
    """
    def __init__(self, max_batt, num_sensors):
        self.max_batt = max_batt
        self.battery_capacity = 2000*3.7#200mAh*3.7V
        self.action_space = spaces.Tuple((spaces.Discrete(1),spaces.Discrete(2)))
        base_state = spaces.Tuple((spaces.Discrete(3),spaces.Discrete(max_batt+1)))
        obs_basis = {'S'+str(i):base_state for i in range(num_sensors)}
        self.observation_space = spaces.Dict(obs_basis)
        self.base_state = {k:(0,max_batt) for k in obs_basis}
        self.state = {k:(0,max_batt) for k in obs_basis}
        self.seed()
        cell_properties = {'system_capacity':2e-3 , 'azimuth':180 , 'tilt':0}
        df = pd.read_pickle('testing.pkl')
        with open('testing.metadata.json') as f:
            meta = {k:converter(v) for k,v in json.load(f).items()}
        generated, dcnet,acnet = solar_getter.convert_to_energy(cell_properties,meta, df)
        self.powerseries = dcnet
        #rendering 
        self.fname = os.getcwd()+'/tmp/'+''.join(random.choice(string.ascii_lowercase) for _ in range(5))+'.json'
        with open(self.fname, 'a+') as f:
            json.dump({'data':[]},f)
        self.record = []
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self, action):
        assert self.action_space.contains(action)
        old_state = self.state
        reward = multi_sensor_env.get_reward(old_state)
        #print('getting reward,{}:{}'.format(state, reward))
        new_state = get_new_state(self.episode_battery_dynamics, 
                                  self.battery_capacity,
                                  self.max_batt,old_state, action)
        #if new_battery == 0:
        #    done=True
        #else: 
        self.state = new_state
        return new_state, reward, False, {}
    def reset(self):
        self.state = self.base_state
        randomstart = random_start_generator()
        currentslice = slicer(self.powerseries, randomstart, self._max_episode_steps)
        randomly_perturbed = list(random_perturber(currentslice).real)
        episode_battery_runner = functools.partial(runner, randomly_perturbed)
        self.episode_battery_dynamics = episode_battery_runner(battery_dynamics)
        #write out record to file for inspection
        with open(self.fname, 'r') as f:
            previous = json.load(f)
        if len(previous['data'])>3:
            previous['data'].pop(0)
        previous['data'].append(self.record)
        with open(self.fname, 'w') as f:
            json.dump(previous,f)
        self.record = []
        return self.state
    def render(self):
        self.record.append(self.state)
