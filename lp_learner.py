import pulp
import numpy as np
import solar_sensor_env
import itertools
import functools
import collections

def cost_calculator(deltat,dutycycle):
    discharge_voltage = 3.7
    on_power = 56+45+15#mAh pyboard plus digimesh plus accel
    off_power = 0.5
    used_power = (on_power*discharge_voltage*deltat*dutycycle)+(off_power*discharge_voltage*deltat*(1-dutycycle))
    return used_power
def generate_constraint(deltat,b0,harvested,duty_vars, t):
    #b1>0
    accumulated = [b0]+[-cost_calculator(deltat,duty_vars[i]) + 1000*deltat*harvested[i] for i in range(t)]
    return accumulated
def battery_transition(deltat, bi, dutycycle, harvestedi):
    consumed = cost_calculator(deltat, dutycycle)
    harvested = 1000*deltat*harvestedi
    return bi+harvested-consumed

def calc_battery(energyseries, dutyseries, deltat, b_initial):
    bnew=b_initial
    batteryseries = [bnew]
    for idx, energy in enumerate(energyseries):
        bprov = battery_transition(deltat,bnew,dutyseries[idx],energy)
        bcapped = capacity if bprov>capacity else bprov
        bnew = bcapped if bcapped>0 else 0
        batteryseries.append(bnew)
    return batteryseries
def solve_eno(deltat,series, b_0):
    problem = pulp.LpProblem('ENO',pulp.LpMaximize)
    times = len(series)
    duty_vars = pulp.LpVariable.dicts("D",[t for t in range(times)],0, 1,'Binary')
    capacity = 2000*3.7 #mAh*3.7V
    problem+=pulp.lpSum(duty_vars)
    #add constraints to ensure nonzero battery at all times
    for i in range(1,times+1):
        b_i = generate_constraint(deltat,b_0,series, duty_vars, i)
        problem += (pulp.lpSum(b_i)>=0,'nonzero batt at t ={}'.format(i))
    #problem += (pulp.lpSum(b_i)<=capacity,'non max batt at t ={}'.format(i))
    #add constraint to ensure energy positive operation
    b_end = generate_constraint(deltat,b_0,series, duty_vars, times)
    problem+=(pulp.lpSum(b_end)>=b_0,'battery at end must be greater than battery at start')
    r = problem.solve()
    assert r==1
    vals = sorted([v for v in problem.variables()], key=lambda i:int(i.name.split('_')[1]))
    dseries = [v.varValue for v in vals]
    batteryseries = calc_battery(series, dseries, deltat,b_0)
    return batteryseries, dseries

def get_series(deltat, numdays, monthrecord, startday):
    deltat = 3
    num_ts = int(24/deltat)
    numdays = 1
    powerseries = solar_sensor_env.downsample(monthrecord, factor=int(48/num_ts))
    times = num_ts*numdays
    series = powerseries[startday*num_ts:startday*num_ts+times]
    return series


class LPAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env,n_episodes=1, max_env_steps = int(365*24/0.5), num_on=1):
        self.env = env
        self.n_episodes = n_episodes
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        self.env.reset()
        self.full_record = []
        self.num_on = num_on
        self.action_plan= {}
    def act(self, observation):
        #recharge if battery gets below 3 
        # status 0: On 1: PreSleep 2: Sleep
        #active_sensors = find_active(observation)
        #print(active_sensors, observation)
        #find min active sensor
        this_t_action = {}
        for sensor, state in observation.items():
        	status, battery, diff,t = state
        	if t==0:
        		#new action plan for a new day
        		#first get the next 8 time steps harvested energy from the oracle
        		globalt = self.env.steps_taken
        		perday = self.env.num_ts
        		mWhbattery = battery*self.env.battery_capacity/self.env.max_batt
        		next_days_solar = self.env.harvested_records[sensor][globalt:globalt+perday]
        		_, dutycycleplan = solve_eno(self.env.deltat, next_days_solar, battery)
        		self.action_plan[sensor] = dutycycleplan
        	action = action_plan[sensor][t]
        	this_t_action[sensor] = action
        return this_t_action
    def run(self, render=True):
        rewards = []
        for e in range(self.n_episodes):
            print('#######New episode#############')
            done=False
            observation = self.env.reset()
            reward_sum = 0
            i=0
            #rewards = []
            while not done and i<self.env._max_episode_steps:
                if render: self.env.render()
                action = self.act(observation)
                #print('observation:{}, action:{}'.format(observation, action))
                observation, reward, done, info = self.env.step(action)
                #print('new observation:{}, reward:{}'.format(observation, reward))
                reward_sum += reward
                i+=1
            rewards.append(reward_sum)
            print("episode: {}/{}, score: {}".format(e, self.n_episodes, reward_sum))
            #self.full_record.append(self.env.record)
        return rewards