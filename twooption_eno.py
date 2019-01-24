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
from solar_sensor_env import (runner, slicer, full_perturber, get_sensor_perturbations,get_generated_power,
                              batt_diff,new_time,might_not_exist_read,downsample,
                              get_random_name) 

def twooption_get_reward(state):
    status, battery = state
    if status ==0 and battery>0:
        reward = 1
    else:
        reward = 0
    return reward
def notdead(state):
    return state[1]>0
def set_initial_status(sensornum):
    return 0 if sensornum=='S0' else 1
def twooption_graph_reward(r_char, n, old_state, sensors):
    """sensors is a list of sensor coords,assumed to be associated in order with the 
    sensor names"""
    active_sensors = [k for k,v in old_state.items() if twooption_get_reward(v[0:2])]
    connectivity = random_graph.is_connected_to_active(sensors, active_sensors, r_char=r_char, n=n)
    hasbatt_sensor_names = [k for k,v in old_state.items() if notdead(v[0:2])]
    connectivity = random_graph.is_connected_to_active(sensors, active_sensors, r_char=r_char, n=n)
    connected = [k for k,v in connectivity.items() if v]
    connected_and_on = [k for k in connected if k in hasbatt_sensor_names]
    capable_rewards =  len(connected_and_on)/(len(sensors))
    #print('active:', active_sensors)
    #print('on: {}, connected: {}, connected_and_on: {}'.format(hasbatt_sensor_names,connected, connected_and_on))
    #print('connectivity {},{}'.format(connectivity, capable_rewards))
    if active_sensors:
        return capable_rewards
    else:
        return -1
badgraphreward = functools.partial(twooption_graph_reward, 12, 3)
goodgraphreward = functools.partial(twooption_graph_reward, 32,1)
def scale_power(time_period, max_batt,energy_generated):
    battery_capacity = 2000*3.7
    max_batt = 10
    power = energy_generated*1000*time_period
    scaled_to_battery = battery*max_batt/battery_capacity
    return scaled_to_battery
def eno_reward(harvested_records, deltat,max_batt, state, state_record, steps_taken):
    #staterecord is like [{'s1':[status, battery,diff,time,'s2: '}, {},{}]
    #harvested is like {"s1":[],'s2'}
    if state['S0'][3]!=0:
        return 0
    else:
        steps_in_one_day = int(24/deltat)
        harvested_start_idx = steps_taken-steps_in_one_day
        harvested_stop_idx = steps_taken
        #
        lastday_energy_harvested = {k:v[harvested_start_idx:harvested_stop_idx] 
                                for k,v in harvested_records.items()}
        lastday_scaled_power = {k:[scale_power(deltat, max_batt, i) for i in v]
                                for k,v in lastday_energy_harvested.items()}
        #
        relevant_states = state_record[-steps_in_one_day:]
        sensornames = [k for k in state]
        states_by_sensor = {k:[state[k] for state in relevant_states]
                            for k in sensornames}
        discrepancies = {k: calculate_discrepancy(state[k],states_by_sensor[k],lastday_scaled_power[k])
                         for k in sensornames}
        vals = [v for k,v in discrepancies.items()]
        average_discrepancy = sum(vals)/len(vals)
        return -1*average_discrepancy
def calculate_discrepancy(state, staterecord, harvested_record):
    #state is like [status, battery, diff, time]
    #calculate consumed from the record
    #record is like [[status, battery, diff, time],....[status, battery, diff, time]]
    starting_batt = staterecord[0][1] 
    finishing_batt = state[1]
    difference = starting_batt - finishing_batt
    total_harvested = sum(harvested_record)
    energy_discrepancy = np.abs(difference-total_harvested)
    return energy_discrepancy
def twooption_battery_dynamics(generated_power,battery_capacity, maxbatt, max_t, status, scaledbattery, randomness=False):
    battery = scaledbattery*(battery_capacity/maxbatt)
    discharge_voltage = 3.7 #Volts
    timeperiod = 48/max_t #hours
    added_power =  generated_power*1000*timeperiod #mWh - generated is avg power in a timeperiod
    
    max_possible = battery_capacity*discharge_voltage #mAh e.g. 2000mAh times 3.7 volts = 7400mW 
    on_power = 56+45+15#mAh pyboard plus digimesh plus accel
    off_power = 45#mAh
    if status == 1:#sleeping
        used_power = (off_power*discharge_voltage*timeperiod)
    else:#either pre-sleep or awake
        used_power = (on_power*timeperiod*discharge_voltage)
    factor = random.normalvariate(1,0.1) if randomness else 1
    balance = added_power - used_power*factor
    new_battery = min(battery+balance, battery_capacity)
    new_battery = max(new_battery,0)
    #print('added power in mWh: {}, used_power: {}, battery:{}, new_battery:{}'.format(added_power, used_power, battery, new_battery))
    normalised = new_battery*(maxbatt/battery_capacity)
    return normalised#int(round(normalised))  

def twooption_status_dynamics(status, battery, action):
    if action == 0:#wakeup
        new_status = 0#awake
    if action == 1:#go to sleep
        new_status =1#sleep
    return new_status

def twooption_get_new_state(batt_funs, battery_capacity, max_batt,max_t, old_state, action):
    """batt_funs is a dictionary of battery functions"""
    action_num,action_val = action
    action_key = 'S'+str(action_num)
    #print('getting reward,{}:{}'.format(state, reward))
    full_actions = {k:multi_sensor_env.what_is_noop(old_state[k]) for k in old_state
                           if k != action_key}
    full_actions[action_key] = action_val

    new_statuses = {k: twooption_status_dynamics(v[0],v[1],full_actions[k])
                        for k,v in old_state.items()}
    new_batteries = {k:batt_funs[k](battery_capacity, max_batt, max_t, v[0],v[1]) for
                         k, v in old_state.items()} 
    diffs = {k:batt_diff(v, old_state[k][1]) for k,v in new_batteries.items()}
    times = {k:new_time(max_t, v[3]) for k,v, in old_state.items()}
    new_state = multi_sensor_env.zip_4dicts(new_statuses, new_batteries, diffs, times)
    #new_state = multi_sensor_env.zip_3dicts(new_statuses, new_batteries, diffs)
    return new_state

class TwoOptionSensorEnv(gym.Env):
    """Simple sleeping sensor environment
    Battery has 10 values, Status has 3 values
    0: On
    1: PreSleep
    2: Sleep
    If the Sensor remains on, you win a reward of 1.
    """
    def __init__(self, max_batt, num_sensors, solarpowerrecord,deltat,
                 recordname=get_random_name(), num_days = 365, 
                 coordinate_generator=random_graph.generate_sorted_network_coords,full_log=False):
        self.max_batt = max_batt
        self.battery_capacity = 2000*3.7#2000mAh*3.7V
        self.action_space = spaces.Tuple((spaces.Discrete(num_sensors),spaces.Discrete(2)))
        self.deltat = deltat
        num_ts = int(24/deltat)
        self.num_ts = num_ts
        self.num_days = num_days
        base_state = spaces.Tuple((spaces.Discrete(2),
                                   spaces.Box(low = np.array([0]), high = np.array([max_batt+1])),
                                   spaces.Box(low = np.array([0]), high = np.array([max_batt+1])),
                                   spaces.Discrete(num_ts)
                                   ))
        self.obs_basis = {'S'+str(i):base_state for i in range(num_sensors)}
        self.coordinate_generator=coordinate_generator
        self.sensors = coordinate_generator(num_sensors)#random_graph.generate_sorted_network_coords(num_sensors)
        self.observation_space = spaces.Dict(self.obs_basis)
        self.state = self.base_state()
        self.seed()
        self.powerseries = downsample(solarpowerrecord, factor=int(48/num_ts))
        #rendering 
        uq = recordname
        self.full_log=full_log
        if full_log:
            self.fname = os.getcwd()+'/tmp/'+uq+'.json'
            self.record = []
        self.rewardfname = os.getcwd()+'/tmp/'+'reward'+uq+'.json'
        self.steps_taken=0
        self.rewards = []
    def base_state(self):
        return {k:(set_initial_status(k),self.max_batt,0,0) for k in self.obs_basis}
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self, action):
        assert self.action_space.contains(action)
        old_state = self.state
        reward = goodgraphreward({k: v[0:2] for k,v 
                                             in old_state.items()},
                             self.sensors)
        #print('getting reward,{}:{}'.format(state, reward))
        #print(old_state, reward)
        new_state = twooption_get_new_state(self.episode_battery_dynamics, 
                                  self.battery_capacity,
                                  self.max_batt,self.num_ts, old_state, action)
        #if new_battery == 0:
        #    done=True
        #else: 
        self.state = new_state
        self.reward = reward
        self.steps_taken+=1
        return new_state, reward, False, {}
    def reset(self):
        self.reward = 0
        self.sensors = self.coordinate_generator(len(self.sensors))#random_graph.generate_network_coords(len(self.sensors))
        randomstart = 0#random_start_generator(self.deltat, self.num_days)
        self.startstep = randomstart
        self.steps_taken = 0
        self.state = self.base_state()
        currentslice = slicer(self.powerseries, randomstart, self._max_episode_steps)
        perturbed = full_perturber(self.sensors, currentslice)
        randomly_perturbed = {k:perturbed[idx]
                              for idx, k in enumerate(self.state)}
        episode_battery_runners = {k:functools.partial(runner, v)
                                  for k,v in randomly_perturbed.items()}
        self.episode_battery_dynamics = {k: episode_battery_runner(twooption_battery_dynamics)
                                         for k, episode_battery_runner 
                                         in episode_battery_runners.items()}
        #write out record to file for inspection
        if self.full_log:
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
            #print('reward: {}'.format(self.rewardfname))
            json.dump(previous_rewards,f)
        self.rewards = []
        return self.state
    def static_initialisation(self, sensors):
        set_status = lambda sensornum, onsensors: 0 if sensornum in onsensors else 2
        self.state = {k:(set_status(k, sensors),self.max_batt,0,0) for k in self.obs_basis}
    def reset_static(self, sensors):
        self.reset()
        self.static_initialisation(sensors)
        print('reset: ', self.state)
        return self.state
    
    def render(self):
        #if self.full_log:
        self.record.append(self.state)
        self.rewards.append(self.reward)


  
class BadTwoOptionSensorEnv(TwoOptionSensorEnv,gym.Env):
    def step(self, action):
        assert self.action_space.contains(action)
        old_state = self.state
        reward = badgraphreward({k: v[0:2] for k,v 
                                             in old_state.items()},
                             self.sensors)
        #print('getting reward,{}:{}'.format(state, reward))
        #print(old_state, reward)
        new_state = twooption_get_new_state(self.episode_battery_dynamics, 
                                  self.battery_capacity,
                                  self.max_batt,self.num_ts, old_state, action)
        #if new_battery == 0:
        #    done=True
        #else: 
        self.state = new_state
        self.reward = reward
        return new_state, reward, False, {}


