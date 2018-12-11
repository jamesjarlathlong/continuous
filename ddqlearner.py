import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Activation, Flatten
from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import itertools
import functools
import multi_sensor_env
from gym.envs.registration import registry, register, make, spec
# Deep Q-learning Agent
def _huber_loss(y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))
def flatten_state(statedict):
    vals = [statedict[k][0::] for k in sorted(statedict)]
    flatvals = list(itertools.chain(*vals))
    return np.reshape(flatvals, [1,len(flatvals)])
def flatten_state_forconv(statedict):
    vals = np.asarray([statedict[k][0::] for k in sorted(statedict)])
    newshape = [1]+list(np.shape(vals))
    return vals.reshape(newshape)
    #flatvals = list(itertools.chain(*vals))
    #return np.reshape(flatvals, [1,len(flatvals)])
def flatten_state_withtime(statedict):
    time = statedict['S0'][3]
    #month = statedict['S0'][4]
    vals = [statedict[k][0:3] for k in sorted(statedict)]
    flatvals = list(itertools.chain(*vals))
    flatvals.append(time)
    #flatvals.append(month)
    return np.reshape(flatvals, [1,len(flatvals)])   
def get_action_size(env):
    action_sizes = [space.n for space in env.action_space.spaces]
    return functools.reduce(lambda x,y:x*y, action_sizes)
def get_alt_action_size(env):
    num_sensors =  env.action_space.spaces[0].n
    return num_sensors + 1
def action_lookup(env, state, action_number):
    lookup = list(itertools.product(*(range(space.n) for space in env.action_space.spaces)))
    return lookup[action_number]
def get_toggle_action(state):
    #0 wakeup, 1 go to sleep
    status = state[0]
    return 1 if status ==0 else 0 #sleep if awake, else wakeup
stringify = lambda num: 'S'+str(num)
def alt_action_lookup(env, state, action_number):
    """state is like{'S0': (0, 10, 0, 0), 'S1': (2, 10, 0, 0)}
    action_number is an int, 0 means toggle sensor 0, 1 toggle sensor 1, -1 noop
    so 0 -> (0, toggle)
    """
    num_actions = get_alt_action_size(env)
    num_sensors = num_actions - 1
    
    relevant_sensor = action_number if action_number < num_sensors else None
    if relevant_sensor!=None:
        sensor_name = stringify(relevant_sensor)
        opposite_action = get_toggle_action(state[sensor_name])
        return (relevant_sensor, opposite_action)
    else:
        #last action is a noop, all others are togglers
        return (0,multi_sensor_env.what_is_noop(state['S0']))


class DDQNAgent:
    def __init__(self, env,n_episodes, max_env_steps=None, modeldir=None, learning_rate = 0.0001, decay_rate = 0.999995, layer_width=64):
        self.env = env
        self.n_episodes = n_episodes
        self.state_size = len(flatten_state_withtime(env.observation_space.sample())[0])
        self.action_size = get_alt_action_size(self.env)
        print(self.action_size, self.state_size)
        self.action_lookup = functools.partial(alt_action_lookup,env)#list(itertools.product(*(range(space.n) for space in env.action_space.spaces)))
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = decay_rate
        self.learning_rate = learning_rate
        self.layer_width = layer_width
        if modeldir:
            self.model = load_model(modeldir,custom_objects = {'_huber_loss':_huber_loss})
        else:
            self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.layer_width, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.layer_width, activation='relu'))
        #model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=_huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model
    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    def replay(self, batch_size):
        minibatch = random.sample(list(self.memory), batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                       np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def run(self, render=True):
        for e in range(self.n_episodes):
            print('#######New episode#############', self.epsilon)
            done=False
            observation = self.env.reset()
            prev_state = flatten_state_withtime(observation)
            reward_sum = 0
            i=0
            while not done and i<self.env._max_episode_steps:
                #print(i)
                if render: self.env.render()
                action = self.act(prev_state)
                # record various intermediates (needed later for backprop)
                # step the environment and get new measurements
                observation, reward, done, info = self.env.step(self.action_lookup(observation, action))
                flat_state = flatten_state_withtime(observation)
                # Remember the previous state, action, reward, and done
                self.remember(prev_state, action, reward, flat_state, done)
                # make next_state the new current state for the next frame.
                prev_state = flat_state
                reward_sum += reward
                i+=1
                if len(self.memory) > 8:
                    #print(i)
                    self.replay(8)
            print("episode: {}/{}, score: {}".format(e, self.n_episodes, reward_sum))
            #agent.replay(32)
            self.update_target_model()
        return e
class DDQNConvAgent:
    def __init__(self, env,n_episodes, max_env_steps=None, modeldir=None, learning_rate = 0.0001, decay_rate = 0.999995, layer_width=64):
        self.env = env
        self.n_episodes = n_episodes
        self.state_size = np.shape(flatten_state_forconv(env.observation_space.sample()))[1::]
        self.action_size = get_alt_action_size(self.env)
        print(self.action_size, self.state_size)
        self.action_lookup = functools.partial(alt_action_lookup,env)#list(itertools.product(*(range(space.n) for space in env.action_space.spaces)))
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = decay_rate
        self.learning_rate = learning_rate
        self.layer_width = layer_width
        if modeldir:
            self.model = load_model(modeldir)
        else:
            self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv1D(self.layer_width, 4, input_shape = self.state_size, padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv1D(self.layer_width, 2, padding='valid'))
        model.add(Activation('relu'))
        #model.add(Dense(128, activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model
    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        #if np.random.rand() <= self.epsilon:
            #return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    def replay(self, batch_size):
        minibatch = random.sample(list(self.memory), batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                shape = np.shape(next_state)
                #shaped = np.reshape(next_state, (1,16,4))
                #print(np.shape(shaped))
                target_pred = self.target_model.predict(next_state)
                target = reward + self.gamma * \
                       np.amax(target_pred[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def run(self, render=True):
        for e in range(self.n_episodes):
            print('#######New episode#############', self.epsilon)
            done=False
            observation = self.env.reset()
            prev_state = flatten_state_forconv(observation)
            reward_sum = 0
            i=0
            while not done and i<self.env._max_episode_steps:
                #print(i)
                if render: self.env.render()
                print(np.shape(prev_state))
                action = self.act(prev_state)
                # record various intermediates (needed later for backprop)
                # step the environment and get new measurements
                observation, reward, done, info = self.env.step(self.action_lookup(observation, action))
                flat_state = flatten_state_forconv(observation)
                # Remember the previous state, action, reward, and done
                self.remember(prev_state, action, reward, flat_state, done)
                # make next_state the new current state for the next frame.
                prev_state = flat_state
                reward_sum += reward
                i+=1
                if len(self.memory) > 8:
                    #print(i)
                    self.replay(8)
            print("episode: {}/{}, score: {}".format(e, self.n_episodes, reward_sum))
            #agent.replay(32)
            self.update_target_model()
        return e
if __name__ == "__main__":
    # initialize gym environment and the agent
    import sys
    nepisodes = int(sys.argv[1])
    fname = sys.argv[2]
    register(
    id='MultiSensor-v0',
    entry_point='multi_sensor_env:MultiSensorEnv',
    kwargs = {'num_sensors':4}
    )
    env = gym.make('MultiSensor-v0')
    agent = DDQNAgent(env,n_episodes = nepisodes, max_env_steps=200)
    agent.run()
    agent.model.save('tmp/{}'.format(fname))
