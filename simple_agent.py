import random

def wrap_action(sensornum, actionnum):
    return (sensornum,actionnum)
class SimpleAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env,n_episodes=10, max_env_steps = int(30*24/0.5)):
        self.env = env
        self.n_episodes = n_episodes
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        self.full_record = []
    def act(self, observation):
        #recharge if battery gets below 3 
        # status 0: On 1: PreSleep 2: Sleep
        #print(observation)
        status, battery, diff = observation['S0']
        if battery<2:
            return wrap_action(0,1)#go to sleep
        elif battery>3:#self.env.max_batt-80:
            #print('maxed', self.env.max_batt, battery)
            return wrap_action(0,0)#wakeup
        else:
            act = 0 if status==0 else 1
            return wrap_action(0,act)
    def run(self, render=True):
        for e in range(self.n_episodes):
            print('#######New episode#############')
            done=False
            observation = self.env.reset()
            reward_sum = 0
            i=0
            while not done and i<self.env._max_episode_steps:
                if render: self.env.render()
                action = self.act(observation)
                #print('observation:{}, action:{}'.format(observation, action))
                observation, reward, done, info = self.env.step(action)
                #print('new observation:{}, reward:{}'.format(observation, reward))
                reward_sum += reward
                i+=1
            print("episode: {}/{}, score: {}".format(e, self.n_episodes, reward_sum))
            #self.full_record.append(self.env.record)
        return e

def find_active(observation):
    #if any sensor is active return the sensorname
    #else return None
    active_sensor = [k for k,v in observation.items() if v[0]==0]
    #assert len(active_sensor)<=1
    return active_sensor
    
class SimpleNetworkAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env,n_episodes=1, max_env_steps = int(365*24/0.5), num_on=1):
        self.env = env
        self.n_episodes = n_episodes
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        self.env.reset()
        self.full_record = []
        self.num_on = num_on
    def act(self, observation):
        #recharge if battery gets below 3 
        # status 0: On 1: PreSleep 2: Sleep
        active_sensors = find_active(observation)
        if len(active_sensors)>=self.num_on:
            #here we check if the battery of any of these is too low
            #and send it to sleep if so
            min_active_sensor = min([(k,v) for k,v
                                     in observation.items()
                                    if k in active_sensors], key = lambda t: t[1][1])
            sensor = min_active_sensor[0]#active_sensors[0]
            sensorname, sensornum = sensor[0], sensor[1::]
            status, battery, diff,t = observation[sensor]
            actionnum = 1 if battery <9 else 0
            wrapped_action = wrap_action(int(sensornum), actionnum)
        else:
            sleepingsensors = [(k,v) for k,v in observation.items() if v[0]==2]
            maxbattsensor = max(sleepingsensors, key = lambda s: s[1][1])
            sensornum = int(maxbattsensor[0].strip('S'))#int(maxbattsensor[0][1])
            #print(sensornum)
            #print('waking up, ',maxbattsensor, sensornum, observation)
            wrapped_action = wrap_action(sensornum, 0)#wakeup
        return wrapped_action
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
class StaticNetworkAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env,on_sensors=None, n_episodes=1, max_env_steps = int(365*24/0.5), num_on=1):
        self.env = env
        self.n_episodes = n_episodes
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        
        self.full_record = []
        if on_sensors:
            self.on_sensors = on_sensors
        else:
            self.on_sensors = ['S'+str(i) for i in range(16) if i==0]
        #self.num_on = num_on
        self.env.reset_static(self.on_sensors)
    def act(self, observation):
        #recharge if battery gets below 3 
        # status 0: On 1: PreSleep 2: Sleep
        active_sensors = find_active(observation)
        #sensor = active_sensors[0]
        #sensorname, sensornum = sensor[0], sensor[1::]
        #action_num = 0
        randomsensor = random.choice([s for s in observation])
        randomaction = random.choice([0,1])
        sensorname, sensornum = randomsensor[0], randomsensor[1::]
        #wrapped_action = wrap_action(int(sensornum), action_num)#wakeup
        wrapped_action = wrap_action(int(sensornum), randomaction)
        #print(wrapped_action)
        return wrapped_action
    def run(self, render=True):
        for e in range(self.n_episodes):
            print('#######New episode#############')
            done=False
            observation = self.env.reset_static(self.on_sensors)
            reward_sum = 0
            i=0
            while not done and i<self.env._max_episode_steps:
                if render: self.env.render()
                action = self.act(observation)
                #print('observation:{}, action:{}'.format(observation, action))
                observation, reward, done, info = self.env.step(action)
                #print('new observation:{}, reward:{}'.format(observation, reward))
                reward_sum += reward
                i+=1
            print("episode: {}/{}, score: {}".format(e, self.n_episodes, reward_sum))
            self.full_record.append(self.env.record)
        return e

class SimpleTwoOptionAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env,n_episodes=1, max_env_steps = int(365*24/0.5), num_on=1):
        self.env = env
        self.n_episodes = n_episodes
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        self.env.reset()
        self.full_record = []
        self.num_on = num_on
    def act(self, observation):
        #recharge if battery gets below 3 
        # status 0: On 1: PreSleep 2: Sleep
        active_sensors = find_active(observation)
        #find min active sensor
        min_active_sensor = min([(k,v) for k,v
                                     in observation.items()
                                    if k in active_sensors], key = lambda t: t[1][1])
        sensor = min_active_sensor[0]#active_sensors[0]
        sensorname, sensornum = sensor[0], sensor[1::]
        status, battery, diff,t = observation[sensor]
        #find max idle sensor
        sleepingsensors = [(k,v) for k,v in observation.items() if v[0]==1]
        maxbattsensor = max(sleepingsensors, key = lambda s: s[1][1])
        sleepingsensor = maxbattsensor[0]
        sleepingname,sleepingnum = sleepingsensor[0], sleepingsensor[1::]
        _,sleeping_batt, _,_  = observation[sleepingsensor]

        if len(active_sensors)<self.num_on:
            wrapped_action = wrap_action(int(sleepingnum),0)#wakeup
        else:
            if battery<sleeping_batt:
                wrapped_action = wrap_action(int(sensornum), 1)#sleep
            else:
                wrapped_action = wrap_action(int(sensornum),0) #noop
        return wrapped_action
    def run(self, render=True):
        rewards = []
        for e in range(self.n_episodes):
            print('#######New episode#############')
            done=False
            observation = self.env.reset()
            reward_sum = 0
            i=0
            
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


class SimpleThreeOptionAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env,n_episodes=1, max_env_steps = int(365*24/0.5), num_on=1):
        self.env = env
        self.n_episodes = n_episodes
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        self.env.reset()
        self.full_record = []
        self.num_on = num_on
    def act(self, observation):
        #recharge if battery gets below 3 
        # status 0: On 1: PreSleep 2: Sleep
        active_sensors = find_active(observation)
        #find min active sensor
        min_active_sensor = min([(k,v) for k,v
                                     in observation.items()
                                    if k in active_sensors], key = lambda t: t[1][1])
        sensor = min_active_sensor[0]#active_sensors[0]
        sensorname, sensornum = sensor[0], sensor[1::]
        status, battery, diff,t = observation[sensor]
        #find max idle sensor
        sleepingsensors = [(k,v) for k,v in observation.items() if v[0] in [1,2]]
        maxbattsensor = max(sleepingsensors, key = lambda s: s[1][1])
        sleepingsensor = maxbattsensor[0]
        sleepingname,sleepingnum = sleepingsensor[0], sleepingsensor[1::]
        _,sleeping_batt, _,_  = observation[sleepingsensor]

        if len(active_sensors)<self.num_on:
            wrapped_action = wrap_action(int(sleepingnum),0)#wakeup
        else:
            if battery<sleeping_batt:
                action = 1 if battery>5 else 2
                wrapped_action = wrap_action(int(sensornum), action)#sleep
            else:
                wrapped_action = wrap_action(int(sensornum),0) #noop as sensornum is in active set
                #wrapped_action = wrap_action(int(sensornum), 2)
        return wrapped_action
    def run(self, render=True):
        for e in range(self.n_episodes):
            print('#######New episode#############')
            done=False
            observation = self.env.reset()
            reward_sum = 0
            i=0
            rewards = []
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
class SimpleThreeOptionAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env,n_episodes=1, max_env_steps = int(365*24/0.5), num_on=1):
        self.env = env
        self.n_episodes = n_episodes
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        self.env.reset()
        self.full_record = []
        self.num_on = num_on
    def act(self, observation):
        #recharge if battery gets below 3 
        # status 0: On 1: PreSleep 2: Sleep
        active_sensors = find_active(observation)
        #print(active_sensors, observation)
        #find min active sensor
        min_active_sensor = min([(k,v) for k,v
                                     in observation.items()
                                    if k in active_sensors], key = lambda t: t[1][1])
        sensor = min_active_sensor[0]#active_sensors[0]
        sensorname, sensornum = sensor[0], sensor[1::]
        status, battery, diff,t = observation[sensor]
        #find max idle sensor
        sleepingsensors = [(k,v) for k,v in observation.items() if v[0] in [1,2]]
        maxbattsensor = max(sleepingsensors, key = lambda s: s[1][1])
        sleepingsensor = maxbattsensor[0]
        sleepingname,sleepingnum = sleepingsensor[0], sleepingsensor[1::]
        _,sleeping_batt, _,_  = observation[sleepingsensor]

        if len(active_sensors)<self.num_on:
            wrapped_action = wrap_action(int(sleepingnum),0)#wakeup
        else:
            if battery<sleeping_batt:
                action = 1 if battery>0 else 2
                wrapped_action = wrap_action(int(sensornum), action)#sleep
            else:
                wrapped_action = wrap_action(int(sensornum),0) #noop as sensornum is in active set
                #wrapped_action = wrap_action(int(sensornum), 2)
        return wrapped_action
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
class SimpleEnoAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env,n_episodes=1, max_env_steps = int(365*24/0.5), num_on=1):
        self.env = env
        self.n_episodes = n_episodes
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        self.env.reset()
        self.full_record = []
        self.num_on = num_on
    def act(self, observation):
        #recharge if battery gets below 3 
        # status 0: On 1: PreSleep 2: Sleep
        #active_sensors = find_active(observation)
        #print(active_sensors, observation)
        #find min active sensor
        sensors = [k for k in observation]
        non50_sensor = [(k,v) for k,v in observation.items()
                            if v[0]!=5]
        sensor = non50_sensor[0][0] if non50_sensor else sensors[0]#active_sensors[0]
        sensorname, sensornum = sensor[0], sensor[1::]
        status, battery, diff,t = observation[sensor]
        #find max idle sensor
        action = 5 if battery>0 else 0
        wrapped_action = wrap_action(int(sensornum), action)#sleep
        return wrapped_action
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