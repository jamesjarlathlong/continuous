

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
        elif battery>4:#self.env.max_batt-80:
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
            self.full_record.append(self.env.record)
        return e

def find_active(observation):
    #if any sensor is active return the sensorname
    #else return None
    active_sensor = [k for k,v in observation.items() if v[0]==0]
    assert len(active_sensor)<=1
    return active_sensor
    
class SimpleNetworkAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env,n_episodes=1, max_env_steps = int(365*24/0.5)):
        self.env = env
        self.n_episodes = n_episodes
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        self.full_record = []
    def act(self, observation):
        #recharge if battery gets below 3 
        # status 0: On 1: PreSleep 2: Sleep
        active_sensors = find_active(observation)
        if active_sensors:
            sensor = active_sensors[0]
            sensorname, sensornum = sensor
            status, battery, diff = observation[sensor]
            actionnum = 1 if battery <99 else 0
            wrapped_action = wrap_action(int(sensornum), actionnum)
        else:
            sleepingsensors = [(k,v) for k,v in observation.items() if v[0]==2]
            maxbattsensor = max(sleepingsensors, key = lambda s: s[1][1])
            sensornum = int(maxbattsensor[0][1])
            wrapped_action = wrap_action(sensornum, 0)#wakeup
        return wrapped_action
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
            self.full_record.append(self.env.record)
        return e
