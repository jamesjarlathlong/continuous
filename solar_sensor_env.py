import gym
from gym import spaces
from gym.utils import seeding
import random_fields
import random_graph
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
import time
import math
import calendar
#850/2344 for simple q learning
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed
def gaussian(x):
    mu = 12
    sig = 2
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
def emulate_sun():
    nighttime = 12*[0.0]
    daytime = [gaussian(0.5*i) for i in range(12,36)]
    return nighttime+daytime+nighttime
def battery_dynamics(generated_power,battery_capacity, maxbatt, max_t, status, scaledbattery):
    battery = scaledbattery*(battery_capacity/maxbatt)
    discharge_voltage = 3.7 #Volts
    timeperiod = 48/max_t #hours
    added_power =  generated_power*1000*timeperiod #mWh - generated is avg power in a timeperiod
    
    max_possible = battery_capacity*discharge_voltage #mAh e.g. 2000mAh times 3.7 volts = 7400mW 
    on_power = 56+45+35#mAh pyboard plus digimesh plus accel
    off_power = 45#mAh
    if status == 2:#sleeping
        used_power = (off_power*discharge_voltage*timeperiod)
    else:#either pre-sleep or awake
        used_power = (on_power*timeperiod*discharge_voltage)
    balance = added_power - used_power
    new_battery = min(battery+balance, battery_capacity)
    new_battery = max(new_battery,0)
    #print('added power in mWh: {}, used_power: {}, battery:{}, new_battery:{}'.format(added_power, used_power, battery, new_battery))
    normalised = new_battery*(maxbatt/battery_capacity)
    return normalised#int(round(normalised))
def runner(reducingseries,f):
    def wrapper(*args, **kwargs):
        current = reducingseries.pop(0)
        return f(current, *args, **kwargs)
    return wrapper
def slicer(wattage, start, duration):
    stop = start+duration
    return list(itertools.islice(itertools.cycle(wattage), start, stop))
def random_perturber(num_sensors, timeseries):
    psd_fun = functools.partial(random_fields.power_spectrum, 100)
    res = []
    for _ in range(num_sensors):
        perturbation = random_fields.gpu_gaussian_proc(psd_fun, size=len(timeseries), scale =150)
        perturbation +=1 #perturbation around self value`
        res.append(list(np.multiply(timeseries, perturbation).real))
    return res
def get_sensor_perturbations(sensors, perturbation):
    return [perturbation[round(i[0]),round(i[1]),:] for i in sensors]
def add_noise(series):
    noise = np.random.normal(0, 0.01, len(series))
    return series+noise
def full_perturber(sensors, timeseries):
    #threedperturbation = random_fields.gpu_gaussian_random_field(size=30,scale=2, length=1)
    #factors = get_sensor_perturbations(sensors, threedperturbation)
    #print(factors)
    factors = [np.array([0]) for _ in sensors]
    res = []
    for f in factors:
        #print('FACTOR: ', f[0])
        fullfactor = np.array([f[0] for _ in timeseries])
        fullfactor+=1
        res.append(list(add_noise(np.multiply(timeseries, fullfactor).real)))
    return res

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
converter = lambda v: float(v) if is_number(v) else v
def rounder(x, base=48):
    return int(base * round(float(x)/base))
def daynumber_to_monthnumber(daynumber):
    monthdays = [calendar.monthrange(2018,i)[1] for i in range(1,13)]
    cumulative = np.cumsum(monthdays)
    cumulative = [0]+list(cumulative)
    monthpairs = zip(cumulative, cumulative[1::])
    return next(idx for idx, el in enumerate(monthpairs) if daynumber in range(*el))

def calculate_month(step_number, steps_per_day):
    day_number = math.floor(step_number/steps_per_day)
    bounded_day_number = day_number%365
    return daynumber_to_monthnumber(bounded_day_number)
def random_start_generator(deltat):
    #deltat = 0.5#hours
    aday = 24/deltat
    num_steps = 365*aday
    return rounder(np.random.randint(0, num_steps), base=aday)
def batt_diff(new_batt, old_batt):
    return new_batt-old_batt
def new_time(max_t, time):
    return time+1 if time<(max_t-1) else 0

def get_new_state(batt_funs, battery_capacity, max_batt,max_t, old_state, action):
    """batt_funs is a dictionary of battery functions"""
    action_num,action_val = action
    action_key = 'S'+str(action_num)
    #print('getting reward,{}:{}'.format(state, reward))
    full_actions = {k:multi_sensor_env.what_is_noop(old_state[k]) for k in old_state
                           if k != action_key}
    full_actions[action_key] = action_val

    new_statuses = {k: sensor_env.status_dynamics(v[0],v[1],full_actions[k])
                        for k,v in old_state.items()}
    new_batteries = {k:batt_funs[k](battery_capacity, max_batt, max_t, v[0],v[1]) for
                         k, v in old_state.items()} 
    diffs = {k:batt_diff(v, old_state[k][1]) for k,v in new_batteries.items()}
    times = {k:new_time(max_t, v[3]) for k,v, in old_state.items()}
    new_state = multi_sensor_env.zip_4dicts(new_statuses, new_batteries, diffs, times)
    #new_state = multi_sensor_env.zip_3dicts(new_statuses, new_batteries, diffs)
    return new_state
def get_new_state_includingtime(batt_funs, battery_capacity, max_batt,
                                max_t,start_step,steps_taken, old_state, action):
    action_num,action_val = action
    action_key = 'S'+str(action_num)
    #print('getting reward,{}:{}'.format(state, reward))
    full_actions = {k:multi_sensor_env.what_is_noop(old_state[k]) for k in old_state
                           if k != action_key}
    full_actions[action_key] = action_val

    new_statuses = {k: sensor_env.status_dynamics(v[0],v[1],full_actions[k])
                        for k,v in old_state.items()}
    new_batteries = {k:batt_funs[k](battery_capacity, max_batt, max_t, v[0],v[1]) for
                         k, v in old_state.items()} 
    diffs = {k:batt_diff(v, old_state[k][1]) for k,v in new_batteries.items()}
    times = {k:new_time(max_t, v[3]) for k,v in old_state.items()}
    months = {k:calculate_month(steps_taken, max_t) for k,v in old_state.items()}
    new_state = multi_sensor_env.zip_5dicts(new_statuses, new_batteries, diffs, times, months)
    #new_state = multi_sensor_env.zip_3dicts(new_statuses, new_batteries, diffs)
    return new_state
    
def might_not_exist_read(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        with open(filename, 'a+') as f:
            data = {'data':[]}
            json.dump(data,f)
    return data
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)
def downsample(series, factor=1):
    avg = lambda x: sum(x)/len(x)
    chunks = grouper(series, factor)
    return list(map(avg, chunks))
def get_generated_power(solarfilename):
    cell_properties = {'system_capacity':2e-3 , 'azimuth':90 , 'tilt':20}
    df = pd.read_pickle(solarfilename+'.pkl')
    with open(solarfilename+'.metadata.json') as f:
        meta = {k:converter(v) for k,v in json.load(f).items()}
        generated, dcnet,acnet = solar_getter.convert_to_energy(cell_properties,meta, df)
    return dcnet
def get_random_name():
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
def set_initial_status(sensornum):
    return 0 if sensornum=='S0' else 2

class SolarSensorEnv(gym.Env):
    """Simple sleeping sensor environment
    Battery has 10 values, Status has 3 values
    0: On
    1: PreSleep
    2: Sleep
    If the Sensor remains on, you win a reward of 1.
    """
    def __init__(self, max_batt, num_sensors, solarpowerrecord,deltat, recordname=get_random_name()):
        self.max_batt = max_batt
        self.battery_capacity = 2000*3.7#2000mAh*3.7V
        self.action_space = spaces.Tuple((spaces.Discrete(num_sensors),spaces.Discrete(2)))
        self.deltat = deltat
        num_ts = int(24/deltat)
        self.num_ts = num_ts
        base_state = spaces.Tuple((spaces.Discrete(3),
                                   spaces.Box(low = np.array([0]), high = np.array([max_batt+1])),
                                   spaces.Box(low = np.array([0]), high = np.array([max_batt+1])),
                                   spaces.Discrete(num_ts)
                                   ))
        self.obs_basis = {'S'+str(i):base_state for i in range(num_sensors)}
        self.sensors = random_graph.generate_network_coords(num_sensors)
        self.observation_space = spaces.Dict(self.obs_basis)
        self.state = self.base_state()
        self.seed()
        self.powerseries = downsample(solarpowerrecord, factor=int(48/num_ts))
        #rendering 
        uq = recordname
        self.fname = os.getcwd()+'/tmp/'+uq+'.json'
        self.rewardfname = os.getcwd()+'/tmp/'+'reward'+uq+'.json'
        self.record = []
        self.rewards = []
    def base_state(self):
        return {k:(set_initial_status(k),max_batt,0,0) for k in self.obs_basis}
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self, action):
        assert self.action_space.contains(action)
        old_state = self.state
        reward = multi_sensor_env.get_reward({k: v[0:2] for k,v 
                                             in old_state.items()})
        #print('getting reward,{}:{}'.format(state, reward))

        new_state = get_new_state(self.episode_battery_dynamics, 
                                  self.battery_capacity,
                                  self.max_batt,self.num_ts, old_state, action)
        #print('action: {}, old_state:{}, new_state: {}'.format(action, old_state, new_state))
        #if new_battery == 0:
        #    done=True
        #else: 
        self.state = new_state
        self.reward = reward
        return new_state, reward, False, {}
    def reset(self):
        self.reward = 0
        self.sensors = random_graph.generate_network_coords(len(self.sensors))
        randomstart = random_start_generator(self.deltat)
        self.startstep = randomstart
        self.steps_taken = 0
        self.state = self.base_state()
        currentslice = slicer(self.powerseries, randomstart, self._max_episode_steps)
        perturbed = full_perturber(self.sensors, currentslice)
        randomly_perturbed = {k:perturbed[idx]
                              for idx, k in enumerate(self.state)}
        episode_battery_runners = {k:functools.partial(runner, v)
                                  for k,v in randomly_perturbed.items()}
        self.episode_battery_dynamics = {k: episode_battery_runner(battery_dynamics)
                                         for k, episode_battery_runner 
                                         in episode_battery_runners.items()}
        #write out record to file for inspection
        previous = might_not_exist_read(self.fname)
        if len(previous['data'])>3:
            previous['data'].pop(0)
        previous['data'].append(self.record)
        with open(self.fname, 'w') as f:
            json.dump(previous,f)
        self.record = []
        previous_rewards = might_not_exist_read(self.rewardfname)
        previous_rewards['data'].append(sum(self.rewards))
        with open(self.rewardfname, 'w') as f:
            print('reward: {}'.format(self.rewardfname))
            json.dump(previous_rewards,f)
        self.rewards = []
        return self.state
    def render(self):
        self.record.append(self.state)
        self.rewards.append(self.reward)

def graph_reward(old_state, sensors):
    """sensors is a list of sensor coords,assumed to be associated in order with the 
    sensor names"""
    active_sensors = [k for k,v in old_state.items() if sensor_env.get_reward(v)]
    connectivity = random_graph.is_connected_to_active(sensors, active_sensors)
    return len([v for k,v in connectivity.items()])/(len(connectivity))
class SolarGraphSensorEnv(SolarSensorEnv, gym.Env):
    def step(self, action):
        assert self.action_space.contains(action)
        old_state = self.state
        reward = multi_sensor_env.get_reward({k: v[0:2] for k,v 
                                             in old_state.items()})
        #print('getting reward,{}:{}'.format(state, reward))
        new_state = get_new_state(self.episode_battery_dynamics, 
                                  self.battery_capacity,
                                  self.max_batt,self.num_ts, old_state, action)
        #if new_battery == 0:
        #    done=True
        #else: 
        self.state = new_state
        self.reward = reward
        return new_state, reward, False, {}
class SolarTimeSensorEnv(SolarSensorEnv, gym.Env):
    def __init__(self, max_batt, num_sensors, solarpowerrecord,deltat, recordname=get_random_name()):
        self.max_batt = max_batt
        self.battery_capacity = 2000*3.7#2000mAh*3.7V
        self.action_space = spaces.Tuple((spaces.Discrete(num_sensors),spaces.Discrete(2)))
        self.deltat = deltat
        num_ts = int(24/deltat)
        self.num_ts = num_ts
        self.num_months = 12
        self.startstep = 0
        base_state = spaces.Tuple((spaces.Discrete(3),
                                   spaces.Box(low = np.array([0]), high = np.array([max_batt+1])),
                                   spaces.Box(low = np.array([0]), high = np.array([max_batt+1])),
                                   spaces.Discrete(num_ts),
                                   spaces.Discrete(self.num_months)
                                   ))
        self.obs_basis = {'S'+str(i):base_state for i in range(num_sensors)}
        self.sensors = random_graph.generate_network_coords(num_sensors)
        self.observation_space = spaces.Dict(self.obs_basis)
        #self.base_state = {k:(set_initial_status(k),max_batt,0,0) for k in obs_basis}
        self.state = self.base_state()
        self.seed()
        self.powerseries = downsample(solarpowerrecord, factor=int(48/num_ts))
        #rendering 
        uq = recordname
        self.fname = os.getcwd()+'/tmp/'+uq+'.json'
        self.rewardfname = os.getcwd()+'/tmp/'+'reward'+uq+'.json'
        self.record = []
        self.rewards = []
    def base_state(self):
        monthnumber = calculate_month(self.startstep,self.num_ts)
        return {k:(set_initial_status(k),self.max_batt,0,0,monthnumber) for k in self.obs_basis}
    def step(self, action):
        assert self.action_space.contains(action)
        old_state = self.state
        reward = multi_sensor_env.get_reward({k: v[0:2] for k,v 
                                             in old_state.items()})
        #print('getting reward,{}:{}'.format(state, reward))

        new_state = get_new_state_includingtime(self.episode_battery_dynamics, 
                                  self.battery_capacity,
                                  self.max_batt,self.num_ts,self.startstep,self.steps_taken, old_state, action)
        #print('action: {}, old_state:{}, new_state: {}'.format(action, old_state, new_state))
        #if new_battery == 0:
        #    done=True
        #else: 
        self.state = new_state
        self.reward = reward
        self.steps_taken+=1
        return new_state, reward, False, {}

